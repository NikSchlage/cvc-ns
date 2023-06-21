/***************************************************************************
 *
 * test_gradient_flow_niko
 *
 ***************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#ifdef _GFLOW_QUDA
#warning "including quda header file quda.h directly "
#include "quda.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#  ifdef HAVE_TMLQCD_LIBWRAPPER
#    include "tmLQCD.h"
#  endif

#ifdef __cplusplus
}
#endif

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "cvc_timer.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "gauge_io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "Q_phi.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "contractions_io.h"
#include "contract_factorized.h"
#include "contract_diagrams.h"
#include "gamma.h"
#include "clover.h"
#include "gradient_flow.h"
#include "gluon_operators.h"
#include "contract_cvc_tensor.h"
#include "scalar_products.h"

using namespace cvc;

/***************************************************************************
 * helper message
 ***************************************************************************/
void usage() {
  fprintf(stdout, "Code for FHT-type nucleon-nucleon 2-pt function inversion and contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  fprintf(stdout, "          -h                  : this message\n");
  EXIT(0);
}




/***************************************************************************
 *
 * MAIN PROGRAM
 *
 ***************************************************************************/
int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "test_gf";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[400];
  struct timeval ta, tb, start_time, end_time;
  int check_propagator_residual = 0;
  unsigned int gf_niter = 10;  /* total number of gradient flow iterations within flow_fwd_gauge_spinor_field */
  double gf_dt = 0.01;         /* small discretization step of flow time */
  unsigned int nsamples = 1;   /* total number of stochastic samples */
  unsigned int nmeas = 1;      /* total number of gradient flow measurements */

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ch?f:e:n:s:m:")) != -1) {
//  while ((c = getopt(argc, argv, "ch?f:e:n:s:m:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 'n':
      gf_niter = atoi ( optarg );
      break;
    case 'e':
      gf_dt = atof ( optarg );
      break;
    case 's':
      nsamples = atoi ( optarg );
      break;
//    case 'm':
//      nmeas = atoi ( optarg );
//      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  gettimeofday ( &start_time, (struct timezone *)NULL );


  /***************************************************************************
   * read input and set the default values
   ***************************************************************************/
  if(filename_set==0) strcpy(filename, "twopt.input");
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_gradient_flow] calling tmLQCD wrapper init functions\n");

  /***************************************************************************
   * initialize tmLQCD solvers
   ***************************************************************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
  if(exitstatus != 0) {
    EXIT(1);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    EXIT(2);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    EXIT(3);
  }
#endif

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /***************************************************************************
   * report git version
   * make sure the version running here has been commited before program call
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_gradient_flow] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_gradient_flow] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_gradient_flow] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_gradient_flow] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[test_gradient_flow] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  
  /***************************************************************************
   * initialize lattice geometry
   *
   * allocate and fill geometry arrays
   ***************************************************************************/
  geometry();


  /***************************************************************************
   * set up some mpi exchangers for
   * (1) even-odd decomposed spinor field
   * (2) even-odd decomposed propagator field
   ***************************************************************************/
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***************************************************************************
   * set up the gauge field
   *
   *   either read it from file or get it from tmLQCD interface
   *
   *   lime format is used
   ***************************************************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [test_gradient_flow] reading gauge field from file %s\n", filename);

    exitstatus = read_lime_gauge_field_doubleprec(filename);

  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test_gradient_flow] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test_gradient_flow] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[test_gradient_flow] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  double *gauge_field_with_phase = NULL;
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  /* exitstatus = gauge_field_eq_gauge_field_ti_bcfactor ( &gauge_field_with_phase, g_gauge_field, -1. ); */
  if(exitstatus != 0) {
    fprintf(stderr, "[test_gradient_flow] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize the clover term, 
   * lmzz and lmzzinv
   *
   *   mzz = space-time diagonal part of the Dirac matrix
   *   l   = light quark mass
   ***************************************************************************/
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  exitstatus = init_clover ( &lmzz, &lmzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_gradient_flow] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[test_gradient_flow] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [test_gradient_flow] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;
  size_t const sizeof_gauge_field = 72 * ( VOLUME ) * sizeof( double );
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );

  /***************************************************************************
   * init rng state
   ***************************************************************************/
  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_gradient_flow] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

#if defined HAVE_LHPC_AFF
  if ( io_proc == 2 ) {
    /***************************************************************************
     * writer for aff output file
     * only I/O process id 2 opens a writer
     ***************************************************************************/
    sprintf(filename, "%s.c%d.aff", outfile_prefix, Nconf );
    fprintf(stdout, "# [test_gradient_flow] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    const char * aff_status_str = aff_writer_errstr ( affw );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_gradient_flow] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  }
#endif




// START CODING FROM HERE:
  for ( unsigned int isample = 0; isample < nsamples; isample++ ) /* loop on isample */
  { 
	  
    /***************************************************************************
     ***************************************************************************
     **
     ** Stochastic source and propagator that are to be flowed
     **
     ***************************************************************************
     ***************************************************************************/

    double ** spinor_field_1 = init_2level_dtable ( 12, _GSI( ( VOLUME+RAND ) ) );
    if ( spinor_field_1 == NULL ) {
      fprintf(stderr, "[test_gradient_flow] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(44);
    }

    /***************************************************************************
     * prepare up-type stoch. propagator from stoch. volume source
     ***************************************************************************/

    double ** spinor_work  = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
    if ( spinor_work == NULL ) {
      fprintf(stderr, "[test_gradient_flow] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(44);
    }

    /* set up a stochstic volume source sw0 */
    exitstatus = prepare_volume_source ( spinor_work[0], VOLUME );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[test_gradient_flow] Error from prepare_volume_source status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /* init sw1 */
    memset ( spinor_work[1], 0, sizeof_spinor_field );

    /* sw1 <- D_up^-1 sw0 */
    exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], 0 );
    if(exitstatus < 0) {
      fprintf(stderr, "[test_gradient_flow] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(44);
    }

    /***************************************************************************
     * copy spinor work field sw0 to spinor field sf1_0,
     * i.e. sf1_0 <- timeslice source
     ***************************************************************************/
    memcpy ( spinor_field_1[0], spinor_work[0], sizeof_spinor_field );    
  
    /***************************************************************************
     * copy spinor work field sw1 to spinor field sf1_1,
     * i.e. sf1_1 <- D_up^-1 sw0
     ***************************************************************************/
    memcpy ( spinor_field_1[1], spinor_work[1], sizeof_spinor_field );

    fini_2level_dtable ( &spinor_work );




    /***************************************************************************
     ***************************************************************************
     **
     ** Gradient Flow (GF) application iteration
     **
     ***************************************************************************
     ***************************************************************************/
#ifdef _GFLOW_CVC  

    /***************************************************************************
     * prepare gauge fields for GF
     ***************************************************************************/
    double * gauge_field_smeared = init_1level_dtable ( 72 * VOLUMEPLUSRAND );
    if ( gauge_field_smeared == NULL ) {
      fprintf(stderr, "[test_gradient_flow] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(3);
    }
  
    memcpy ( gauge_field_smeared, gauge_field_with_phase, sizeof_gauge_field );
    
    double * gauge_field_smeared_2 = init_1level_dtable ( 72 * VOLUMEPLUSRAND );
    if ( gauge_field_smeared_2 == NULL ) {
      fprintf(stderr, "[test_gradient_flow] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(3);
    }
  
    memcpy ( gauge_field_smeared_2, gauge_field_smeared, sizeof_gauge_field );
      
    /***************************************************************************
     * prepare spinor fields for GF
     ***************************************************************************/
    double ** spinor_field_2  = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
    if ( spinor_field_2 == NULL ) {
      fprintf(stderr, "[test_gradient_flow] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(44);
    }
  
    /* init sf2_0 and sf2_1 */
    memset ( spinor_field_2[0], 0, sizeof_spinor_field );
    memset ( spinor_field_2[1], 0, sizeof_spinor_field );
  
  
    double ** spinor_field_3  = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
    if ( spinor_field_3 == NULL ) {
      fprintf(stderr, "[test_gradient_flow] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(44);
    }
  
    /* init sf3_0 and sf3_1 */
    memset ( spinor_field_3[0], 0, sizeof_spinor_field );
    memset ( spinor_field_3[1], 0, sizeof_spinor_field );
  

    /***************************************************************************
     * prepare complex numbers
     ***************************************************************************/
    complex *zchi_aux = ( complex* ) malloc ( sizeof ( complex ) );
    complex *zchi_aux2 = ( complex* ) malloc ( sizeof ( complex ) );
    complex *zchi = ( complex* ) malloc ( sizeof ( complex ) );
  

    /***************************************************************************
     * prepare Dirac gamma matrices
     * cf. gamma.cpp: gamma_matrix_set( gamma_matrix_type *g, int id, double s )
     ***************************************************************************/
     
    /*=====================================================*/
    /*  vector (v) :                                       */
    /*  gamma_0 = gamma_t         :    igamma = 0, ig = 0  */
    /*  gamma_1 = gamma_x         :    igamma = 0, ig = 1  */
    /*  gamma_2 = gamma_          :    igamma = 0, ig = 2  */
    /*  gamma_3 = gamma_z         :    igamma = 0, ig = 3  */
    
    /*  pseudovector (pv) :                                */
    /*  gamma_6 = gamma_5 gamma_t :    igamma = 1, ig = 0  */
    /*  gamma_7 = gamma_5 gamma_x :    igamma = 1, ig = 1  */
    /*  gamma_8 = gamma_5 gamma_y :    igamma = 1, ig = 2  */
    /*  gamma_9 = gamma_5 gamma_z :    igamma = 1, ig = 3  */

    /*  scalar (s) :                                       */
    /*  gamma_4 = id              :    igamma = 2, ig = 0  */

    /*  pseudoscalar (ps) :                                */
    /*  gamma_5                   :    igamma = 3, ig = 0  */
    /*=====================================================*/
    
    int const gamma_id[4][4] = {            /* gamma_id[igamma][ig] */
      { 0,  1,  2,  3 },
      { 6,  7,  8,  9 },
      { 4, -1, -1, -1 },
      { 5, -1, -1, -1 } };
     
    int const gamma_sets = 1;               /* only use gamma_0, gamma_1, gamma_2 and gamma_3 in for loop on Gamma structures;
                                               hence, only igamma = 0 => gamma_sets = 1, cf. for loop on Gamma structures */
    int const gamma_num[4] = {4, 4, 1, 1};  /* {4, 4, 1, 1} because in 1st and 2nd row of gamma_id all entries (i.e. ig)
                                               are relevant and in 3rd and 4th row only the first entry being the index
                                               of the gamma_4 and gamma_5, respectively */
  
  
  

    /***************************************************************************
     ***************************************************************************
     **
     ** GF iterations
     **
     ***************************************************************************
     ***************************************************************************/
    int gf_niter_tst[20] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5 };
    double gf_dt_tst[20] = { 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 };
    
    double gf_t = 0.;
    unsigned int nmeas = sizeof(gf_niter_tst) / sizeof(gf_niter_tst[0]);
    unsigned int nmeas2 = sizeof(gf_dt_tst) / sizeof(gf_dt_tst[0]);
    
    if ( nmeas / nmeas2  != 1 ) {
      fprintf(stderr, "[test_gradient_flow] Error, size of gf_niter and gf_dt must be equal!", __FILE__, __LINE__ );
      EXIT(44);
    }
    
     
    for ( unsigned int i = 0; i < nmeas; i++ )
    {
		
        /* flow timeslice source sf1_0 and gauge field */
        gettimeofday ( &ta, (struct timezone *)NULL );

        flow_fwd_gauge_spinor_field ( gauge_field_smeared, spinor_field_1[0], gf_niter_tst[i], gf_dt_tst[i], 1, 1 ); // returns flowed version of spinor_field_1[0] and of gauge_field_smeared

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "test_gradient_flow", "flow_fwd_gauge_spinor_field", g_cart_id == 0 );
      
        /* flow propagator sf1_1 <- D_up^-1 sw0 and gauge field */
        gettimeofday ( &ta, (struct timezone *)NULL );

        flow_fwd_gauge_spinor_field ( gauge_field_smeared_2, spinor_field_1[1], gf_niter_tst[i], gf_dt_tst[i], 1, 1 ); // returns flowed version of spinor_field_1[1] and of gauge_field_smeared

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "test_gradient_flow", "flow_fwd_gauge_spinor_field", g_cart_id == 0 );


      /***************************************************************************
       ***************************************************************************
       **
       ** Calculation of normalization Z_chi
       **
       ***************************************************************************
       ***************************************************************************/
       
      _co_eq_zero( zchi_aux2 );
  
      /***************************************************************************
       * loop on Gamma structures (v, pv, s, ps) and gamma matrix indices mu
       ***************************************************************************/
      for ( int igamma = 0; igamma < gamma_sets; igamma++ )  /* loop on Gamma structures; only use gamma_0, gamma_1, gamma_2 and gamma_3 here; hence, only igamma = 0 => gamma_sets = 1 */
      {
        for ( int ig = 0; ig < gamma_num[igamma]; ig++ ) {  /* loop on Dirac gamma matrix indices mu */
          /***************************************************************************
           * calculate fwd/ bwd displacement for construction of covariant derivative:
           * spinor_field_eq_cov_displ_spinor_field ( double * const s, double * const r_in, int const mu, int const fbwd, double * const gauge_field )
           * from Q_phi.cpp
           ***************************************************************************/
          /* forward (fbwd = 0)*/
          spinor_field_eq_cov_displ_spinor_field ( spinor_field_2[0], spinor_field_1[1], ig, 0, gauge_field_smeared );
          /* backward (fbwd = 1)*/
          spinor_field_eq_cov_displ_spinor_field ( spinor_field_2[1], spinor_field_1[1], ig, 1, gauge_field_smeared );
      
          /***************************************************************************
           * calculate covariant derivative of propagator sf3_0 <- D sf1_1:
           * spinor_field_eq_spinor_field_mi_spinor_field(double*r, double*s, double*t, unsigned int N)
           * from cvc_utils.cpp, i.e "r = s - t"
           ***************************************************************************/
          spinor_field_eq_spinor_field_mi_spinor_field( spinor_field_3[0], spinor_field_2[0], spinor_field_2[1], VOLUME );
      
          /***************************************************************************
           * Part I - calculate zchi_aux2 <- sf1_0^dag gamma D sf1_1:
           * calculate sf3_1 <- gamma D sf1_1 using
           * spinor_field_eq_gamma_ti_spinor_field(double*r, int gid, double*s, unsigned int N)
           * from cvc_utils.cpp
           ***************************************************************************/
          spinor_field_eq_gamma_ti_spinor_field( spinor_field_3[1], gamma_id[igamma][ig], spinor_field_3[0], VOLUME );
      
          /***************************************************************************
           * Part II - calculate zchi_aux2 <- sf1_0^dag gamma D sf1_1:
           * calculate scalar product for spinor fields, i.e. sf1_0^dag gamma D sf1_1, using
           * spinor_scalar_product_co(complex *w, double *xi, double *phi, unsigned int V)
           * from scalar_products.cpp
           ***************************************************************************/
          spinor_scalar_product_co( zchi_aux, spinor_field_1[0], spinor_field_3[1], VOLUME );
      
          /***************************************************************************
           * Part III - calculate Dirac trace zchi_aux2 <- sf1_0^dag gamma D sf1_1:
           * sum up zchi_aux contributions using
           * _co_pl_eq_co(c1,c2)
           * defined in cvc_complex.h
           ***************************************************************************/
          _co_pl_eq_co( zchi_aux2, zchi_aux );
      
        } /* end of loop on Dirac gamma matrix indices mu */
      } /* end of loop on Gamma structures */
      
  
      /***************************************************************************
       * Due to closed fermion loop Wick contraction, multiply with factor -1:
       * zchi <- (-1) * zchi_aux2
       ***************************************************************************/
      _co_eq_re_by_co( zchi, -1, zchi_aux2 );
  
      
      //fprintf( stdout, "# [test_gradient_flow] isample = %d; gf_t = %f; Zchi_re = %f; Zchi_im = %f\n", isample, gf_t, zchi->re, zchi->im );
      fprintf( stdout, "%d %f %f %f\n", isample, gf_t, zchi->re, zchi->im );
      
      gf_t += gf_niter_tst[i] * gf_dt_tst[i];  /* flowtime is given by gf_t += gf_niter[i]*gf_dt[i] */
  
    }  /* end of loop on i */

    /***************************************************************************
     * deallocate gf gauge fields
     ***************************************************************************/
    fini_1level_dtable ( &gauge_field_smeared );
    fini_1level_dtable ( &gauge_field_smeared_2 );
      
    /***************************************************************************
     * deallocate gf spinor fields
     ***************************************************************************/
    fini_2level_dtable ( &spinor_field_2 );
    fini_2level_dtable ( &spinor_field_3 );

    free( zchi_aux );
    free( zchi_aux2 );
    free( zchi );
  
#endif  /* of if def _GFLOW_CVC */


    /***************************************************************************
     * deallocate (static) fields
     ***************************************************************************/
    exitstatus = prepare_volume_source ( NULL, 0 );
    fini_2level_dtable ( &spinor_field_1 );
    flow_fwd_gauge_spinor_field ( NULL, NULL, 0, 0., 0, 0 );

  }  /* end of loop on isample */
  /***************************************************************************/
  /***************************************************************************/




#ifdef HAVE_LHPC_AFF
  /***************************************************************************
   * I/O process id 2 closes its AFF writer
   ***************************************************************************/
  if(io_proc == 2) {
    const char * aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_gradient_flow] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

  /***************************************************************************
   * free the allocated memory, finalize
  ***************************************************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  if ( g_gauge_field != NULL ) free(g_gauge_field);
#endif
  if ( gauge_field_with_phase != NULL ) free ( gauge_field_with_phase );

  /* free clover matrix terms */
  fini_clover ( );

  /* free lattice geometry arrays */
  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "test_gradient_flow", "runtime", g_cart_id == 0 );

  return(0);

}
