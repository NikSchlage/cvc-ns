/***************************************************************************
 *
 * njjn_fht_invert_contract
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
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "contractions_io.h"
#include "Q_clover_phi.h"
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

#define _OP_ID_UP 0
#define _OP_ID_DN 1

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
  
  const char outfile_prefix[] = "njjn_fht";

  const int sequential_gamma_sets = 4;

  int const sequential_gamma_num[4] = {4, 4, 1, 1};

  char const sequential_gamma_tag[4][3] = { "vv", "aa", "ss", "pp" };

  char const gamma_id_to_Cg_ascii[16][10] = {
    "Cgy",
    "Cgzg5",
    "Cgt",
    "Cgxg5",
    "Cgygt",
    "Cgyg5gt",
    "Cgyg5",
    "Cgz",
    "Cg5gt",
    "Cgx",
    "Cgzg5gt",
    "C",
    "Cgxg5gt",
    "Cgxgt",
    "Cg5",
    "Cgzgt"
  };


  char const gamma_id_to_ascii[16][10] = {
    "gt",
    "gx",
    "gy",
    "gz",
    "id",
    "g5",
    "gtg5",
    "gxg5",
    "gyg5",
    "gzg5",
    "gtgx",
    "gtgy",
    "gtgz",
    "gxgy",
    "gxgz",
    "gygz" 
  };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[100];
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  double *gauge_field_smeared = NULL;
  gamma_matrix_type sequential_gamma_list[4][4];

  int const    gamma_f1_number                           = 4;
  int const    gamma_f1_list[gamma_f1_number]            = { 14 , 11,  8,  2 };
  double const gamma_f1_sign[gamma_f1_number]            = { +1 , +1, -1, -1 };

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char aff_tag[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  g_the_time = time(NULL);

  /***************************************************************************
   * read input and set the default values
   ***************************************************************************/
  if(filename_set==0) strcpy(filename, "twopt.input");
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [njjn_fht_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [njjn_fht_invert_contract] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [njjn_fht_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [njjn_fht_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[njjn_fht_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[njjn_fht_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [njjn_fht_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [njjn_fht_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[njjn_fht_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[njjn_fht_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[njjn_fht_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * if we want to use Jacobi smearing, we need 
   * smeared gauge field
   ***********************************************/
  if( N_Jacobi > 0 ) {

    alloc_gauge_field ( &gauge_field_smeared, VOLUMEPLUSRAND);

    memcpy ( gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));

    if ( N_ape > 0 ) {
      exitstatus = APE_Smearing(gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
    }  /* end of if N_aoe > 0 */
  }  /* end of if N_Jacobi > 0 */

  /***************************************************************************
   * initialize the clover term, 
   * lmzz and lmzzinv
   *
   *   mzz = space-time diagonal part of the Dirac matrix
   *   l   = light quark mass
   ***************************************************************************/
  exitstatus = init_clover ( &lmzz, &lmzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[njjn_fht_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[njjn_fht_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [njjn_fht_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * set the gamma matrices
   ***************************************************************************/
  /* vector */
  gamma_matrix_set( &( sequential_gamma_list[0][0] ), 0, 1. );  /*  gamma_0 = gamma_t */
  gamma_matrix_set( &( sequential_gamma_list[0][1] ), 1, 1. );  /*  gamma_1 = gamma_x */
  gamma_matrix_set( &( sequential_gamma_list[0][2] ), 2, 1. );  /*  gamma_2 = gamma_y */
  gamma_matrix_set( &( sequential_gamma_list[0][3] ), 3, 1. );  /*  gamma_3 = gamma_z */
  /* pseudovector */
  gamma_matrix_set( &( sequential_gamma_list[1][0] ), 6, 1. );  /*  gamma_6 = gamma_5 gamma_t */
  gamma_matrix_set( &( sequential_gamma_list[1][1] ), 7, 1. );  /*  gamma_7 = gamma_5 gamma_x */
  gamma_matrix_set( &( sequential_gamma_list[1][2] ), 8, 1. );  /*  gamma_8 = gamma_5 gamma_y */
  gamma_matrix_set( &( sequential_gamma_list[1][3] ), 9, 1. );  /*  gamma_9 = gamma_5 gamma_z */
  /* scalar */
  gamma_matrix_set( &( sequential_gamma_list[2][0] ), 4, 1. );  /*  gamma_4 = id */
  /* pseudoscalar */
  gamma_matrix_set( &( sequential_gamma_list[3][0] ), 5, 1. );  /*  gamma_5 */


  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );

  double _Complex ** ephase = NULL;
  if ( g_seq_source_momentum_number > 0 ) {
    ephase = init_2level_ztable ( g_seq_source_momentum_number, VOL3 );
    if ( ephase == NULL ) {
      fprintf ( stderr, "[njjn_fht_invert_contract] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }

    make_phase_field_timeslice ( ephase, g_seq_source_momentum_number, g_seq_source_momentum_list );
  }  /* end of if g_seq_source_momentum_number > 0 */

  /***************************************************************************
   * init rng state
   ***************************************************************************/
  exitstatus = init_rng_state ( g_seed, &g_rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[njjn_fht_invert_contract] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***************************************************************************
   *
   * prepare stochastic propagators
   *
   ***************************************************************************/
  double _Complex *** loop = init_3level_ztable ( VOLUME, 12, 12 );
  if ( loop  == NULL ) {
    fprintf ( stderr, "[njjn_fht_invert_contract] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }

  /***************************************************************************
   * loop on samples
   ***************************************************************************/
  for ( int isample = 0; isample < g_nsample; isample++ ) {

    double * stochastic_source = init_1level_dtable ( _GSI( VOLUME ) );
    if ( stochastic_source == NULL ) {
      fprintf ( stderr, "[njjn_fht_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
 
    double ** spinor_work = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
    if ( spinor_work == NULL ) {
      fprintf ( stderr, "[njjn_fht_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }

    exitstatus = prepare_volume_source ( stochastic_source, VOLUME );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[njjn_fht_invert_contract] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(12);
    }

    for ( int ispin = 0; ispin < 4; ispin++ ) {
      for ( int icol = 0; icol < 3; icol++ ) {

        int const isc = 3 * ispin + icol;

#pragma omp parallel for
        for ( unsigned int ix = 0; ix < VOLUME; ix++  ) {
          spinor_work[0][_GSI(ix)+2*isc  ] = stochastic_source[_GSI(ix)+2*isc  ];
          spinor_work[0][_GSI(ix)+2*isc+1] = stochastic_source[_GSI(ix)+2*isc+1];
        }

        /* tm-rotate stochastic propagator at source */
        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation(spinor_work[0], spinor_work[0], 1, g_fermion_type, VOLUME);
        }

        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], _OP_ID_UP );
        if(exitstatus != 0) {
          fprintf(stderr, "[njjn_fht_invert_contract] Error from tmLQCD_invert, status was %d\n", exitstatus);
          EXIT(12);
        }

        if ( check_propagator_residual ) {
          check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, lmzz[_OP_ID_UP], 1 );
        }

        /* tm-rotate stochastic propagator at sink */
        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation(spinor_work[1], spinor_work[1], 1, g_fermion_type, VOLUME);
        }

#pragma omp parallel for
        for ( unsigned int ix = 0; ix < VOLUME; ix++  ) {
          for ( int kspin = 0; kspin < 4; kspin++ ) {
            for ( int kcol = 0; kcol < 3; kcol++ ) {
              int const ksc = 3 * kspin + kcol;

              loop[ix][ksc][isc] +=
                  ( stochastic_source[_GSI(ix) + 2*isc  ] + I * stochastic_source[_GSI(ix) + 2*isc+1] )
                * ( spinor_work[1][_GSI(ix)    + 2*ksc  ] + I * spinor_work[1][_GSI(ix)    + 2*ksc+1] );
            }
          }
        }
      }  /* end of loop on color dilution component */
    }  /* end of loop on spin dilution component */

    fini_1level_dtable ( &stochastic_source );
    fini_2level_dtable ( &spinor_work );
  }  /* end of loop on samples */


  /***************************************************************************
   * normalize
   ***************************************************************************/
#pragma omp parallel for
  for ( unsigned int ix = 0; ix < 144 * VOLUME; ix++  ) {
    loop[0][0][ix] /= (double)g_nsample;
  }


  /***************************************************************************
   *
   * point-to-all version
   *
   ***************************************************************************/

  /***************************************************************************
   * loop on source locations
   *
   *   each source location is given by 4-coordinates in
   *   global variable
   *   g_source_coords_list[count][0-3] for t,x,y,z
   ***************************************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

    /***************************************************************************
     * allocate point-to-all propagators
     ***************************************************************************/

    /* up quark propagator with source and sink smearing*/
    double ** propagator_up = init_2level_dtable ( 12, _GSI( VOLUME ) );
    if( propagator_up == NULL ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /* up quark propagator with sink smearing to use for baryon 2-pt function */
    double ** propagator_up_snk_smeared = init_2level_dtable ( 12, _GSI( VOLUME ) );
    if ( propagator_up_snk_smeared == NULL ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /* down quark propagator */
    double ** propagator_dn = init_2level_dtable ( 12, _GSI( VOLUME ) );
    if( propagator_dn == NULL ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /* down quark propagator with sink smearing to use for baryon 2-pt function */
    double ** propagator_dn_snk_smeared = init_2level_dtable ( 12, _GSI( VOLUME ) );
    if( propagator_dn_snk_smeared == NULL ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /***********************************************************
     * determine source coordinates, find out, if source_location is in this process
     ***********************************************************/

    int const gsx[4] = {
        ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global,
        ( g_source_coords_list[isource_location][1] + LX_global ) % LX_global,
        ( g_source_coords_list[isource_location][2] + LY_global ) % LY_global,
        ( g_source_coords_list[isource_location][3] + LZ_global ) % LZ_global };

    int sx[4], source_proc_id = -1;
    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /***********************************************
     * open output file reader
     * we use the AFF format here
     * https://github.com/usqcd-software/aff
     ***********************************************/
#if ( defined HAVE_LHPC_AFF )
    /***********************************************
     * writer for aff output file
     * only I/O process id 2 opens a writer
     ***********************************************/
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [njjn_fht_invert_contract] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#else
    fprintf(stderr, "[njjn_fht_invert_contract] Error, no AFF lib %s %d\n",  __FILE__, __LINE__);
    EXIT(15);
#endif

    /**********************************************************
     *
     * point-to-all propagators with source at gsx
     *
     **********************************************************/

    /***********************************************************
     * up-type point-to-all propagator
     *
     * ONLY SOURCE smearing here
     *
     * NOTE: quark flavor is controlled by 
     * _OP_ID_UP and _OP_ID_DN below
     ***********************************************************/
    exitstatus = point_source_propagator ( propagator_up, gsx, _OP_ID_UP, 1, 0, gauge_field_smeared, check_propagator_residual, gauge_field_with_phase, lmzz );
    if(exitstatus != 0) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(12);
    }

    /***********************************************************
     * sink-smear the up-type point-to-all propagator
     ***********************************************************/
    for ( int i = 0; i < 12; i++ ) {
      /* copy propagator */
      memcpy ( propagator_up_snk_smeared[i], propagator_up[i], sizeof_spinor_field );

      /* sink-smear propagator */
      exitstatus = Jacobi_Smearing ( gauge_field_smeared, propagator_up_snk_smeared[i], N_Jacobi, kappa_Jacobi);
      if(exitstatus != 0) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        return(11);
      }
    }

    /***********************************************************
     * optionally write the propagator to disc
     *
     * we use the standard lime format here
     * https://github.com/usqcd-software/c-lime
     ***********************************************************/
    if ( g_write_propagator ) {
      /* each spin-color component into a separate file */
      for ( int i = 0; i < 12; i++ ) {
        sprintf ( filename, "propagator_up.%.4d.t%dx%dy%dz%d.%d.inverted", Nconf, gsx[0], gsx[1], gsx[2], gsx[3] , i );

        if ( ( exitstatus = write_propagator( propagator_up[i], filename, 0, g_propagator_precision) ) != 0 ) {
          fprintf(stderr, "[njjn_fht_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }
    }

    /***********************************************************
     * dn-type point-to-all propagator
     *
     * ONLY SOURCE smearing here
     ***********************************************************/
    exitstatus = point_source_propagator ( propagator_dn, gsx, _OP_ID_DN, 1, 0, gauge_field_smeared, check_propagator_residual, gauge_field_with_phase, lmzz );
    if(exitstatus != 0) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(12);
    }

    /***********************************************************
     * sink-smear the dn-type point-to-all propagator
     ***********************************************************/
    for ( int i = 0; i < 12; i++ ) {
      /* copy propagator */
      memcpy ( propagator_dn_snk_smeared[i], propagator_dn[i], sizeof_spinor_field );

      /* sink-smear propagator */
      exitstatus = Jacobi_Smearing ( gauge_field_smeared, propagator_dn_snk_smeared[i], N_Jacobi, kappa_Jacobi);
      if(exitstatus != 0) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        return(11);
      }
    }

    if ( g_write_propagator ) {
      for ( int i = 0; i < 12; i++ ) {
        sprintf ( filename, "propagator_dn.%.4d.t%dx%dy%dz%d.%d.inverted", Nconf, 
            gsx[0], gsx[1], gsx[2], gsx[3] , i );

        if ( ( exitstatus = write_propagator( propagator_dn[i], filename, 0, g_propagator_precision) ) != 0 ) {
          fprintf(stderr, "[njjn_fht_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }
    }

    /***************************************************************************
     *
     * contractions for the nucleon 2-point functions
     *
     * now we have up and down propagator and can proceed with
     * contractions
     * only 2 diagrams here, n1 and n2
     *
     ***************************************************************************/


    /***************************************************************************
     * allocate propagator fields
     *
     * these are ordered as as
     * t,x,y,z,spin,color,spin,color
     * so a 12x12 complex matrix per space-time point
     ***************************************************************************/
    fermion_propagator_type * fp  = create_fp_field ( VOLUME );
    fermion_propagator_type * fp2 = create_fp_field ( VOLUME );
    fermion_propagator_type * fp3 = create_fp_field ( VOLUME );


    /***************************************************************************
     * vx holds the x-dependent nucleon-nucleon spin propagator,
     * i.e. a 4x4 complex matrix per space time point
     ***************************************************************************/
    double ** vx = init_2level_dtable ( VOLUME, 32 );
    if ( vx == NULL ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
      EXIT(47);
    }

    /***************************************************************************
     * vp holds the nucleon-nucleon spin propagator in momentum space,
     * i.e. the momentum projected vx
     ***************************************************************************/
    double *** vp = init_3level_dtable ( T, g_sink_momentum_number, 32 );
    if ( vp == NULL ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(47);
    }

    /***************************************************************************
     *
     * spin 1/2 2-point correlation function
     *
     ***************************************************************************/

    char  aff_tag_prefix[100];
    sprintf ( aff_tag_prefix, "/N-N/T%d_X%d_Y%d_Z%d", gsx[0], gsx[1], gsx[2], gsx[3] );
         
    /***************************************************************************
     * fill the fermion propagator fp with the 12 spinor fields
     * in propagator_up
     ***************************************************************************/
    assign_fermion_propagator_from_spinor_field ( fp, propagator_up_snk_smeared, VOLUME);

    /***************************************************************************
     * fill fp2 with 12 spinor fields from propagator_dn
     ***************************************************************************/
    assign_fermion_propagator_from_spinor_field ( fp2, propagator_dn_snk_smeared, VOLUME);

    /***************************************************************************
     * contractions for proton - proton
     *
     * if1/2 loop over various Dirac Gamma-structures for
     * nucleon interpolators at source and sink
     ***************************************************************************/
    for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
    for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {

      /***************************************************************************
       * here we calculate fp3 = Gamma[if2] x propagator_dn / fp2 x Gamma[if1]
       ***************************************************************************/
      fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp2, VOLUME );

      fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );

      fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );

      /***************************************************************************
       * diagram n1
       *
       * a name for the data set
       ***************************************************************************/
      sprintf(aff_tag, "%s/Gi_%s/Gf_%s/n1", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

      /* the actual contraction
       *   the operation is called type v5 here  */
      exitstatus = contract_v5 ( vx, fp, fp3, fp, VOLUME );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v5, status was %d\n", exitstatus);
        EXIT(48);
      }

      /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
      exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
        EXIT(48);
      }

      /* write to AFF file */
      exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
        EXIT(49);
      }

      /***************************************************************************
       * diagram n2
       ***************************************************************************/
      sprintf(aff_tag, "%s/Gi_%s/Gf_%s/n2", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

      exitstatus = contract_v6 ( vx, fp, fp3, fp, VOLUME );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v6, status was %d\n", exitstatus);
        EXIT(48);
      }

      exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
        EXIT(48);
      }

      exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
        EXIT(49);
      }

    }}  /* end of loop on Dirac Gamma structures */
    
    /***************************************************************************
     * contractions for neutron neutron
     *
     * if1/2 loop over various Dirac Gamma-structures for
     * nucleon interpolators at source and sink
     ***************************************************************************/
    for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
    for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {

      /***************************************************************************
       * here we calculate fp3 = Gamma[if2] x propagator_up / fp x Gamma[if1]
       ***************************************************************************/
      fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp,  VOLUME );

      fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );

      fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );

      /***************************************************************************
       * diagram n3
       ***************************************************************************/
      sprintf ( aff_tag, "%s/Gi_%s/Gf_%s/n3", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

      /* the actual contraction
       *   the operation is called type v5 here  */
      exitstatus = contract_v5 ( vx, fp2, fp3, fp2, VOLUME );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v5, status was %d\n", exitstatus);
        EXIT(48);
      }

      /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
      exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
        EXIT(48);
      }

      /* write to AFF file */
      exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
        EXIT(49);
      }

      /***************************************************************************
       * diagram n4
       ***************************************************************************/
      sprintf ( aff_tag, "%s/Gi_%s/Gf_%s/n4", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

      exitstatus = contract_v6 ( vx, fp2, fp3, fp2, VOLUME );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v6, status was %d\n", exitstatus);
        EXIT(48);
      }

      exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
        EXIT(48);
      }

      exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
        EXIT(49);
      }

      /***************************************************************************
       * done
       ***************************************************************************/
    }}  /* end of loop on Dirac Gamma structures */


    /***************************************************************************
     *
     * sequential inversion and contraction
     *
     ***************************************************************************/

    /* allocate sequential propagator */
    double ** sequential_propagator = init_2level_dtable ( 12, _GSI( VOLUME ) );
    if( sequential_propagator == NULL ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /***************************************************************************
     * loop on sequential source momenta
     ***************************************************************************/
    for ( int imom = 0; imom < g_seq_source_momentum_number; imom++ ) {

      double *** sequential_source = init_3level_dtable ( 2, 12,  _GSI(VOLUME) );
      if( sequential_source == NULL ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(132);
      }

      int momentum[3] = {
          g_seq_source_momentum_list[imom][0],
          g_seq_source_momentum_list[imom][1],
          g_seq_source_momentum_list[imom][2] };

      /***************************************************************************
       ***************************************************************************
       **
       ** Part II: sequential up - after - up inversion and contraction
       **
       ***************************************************************************
       ***************************************************************************/

      /***************************************************************************
       * loop on sequential source gamma matrices
       ***************************************************************************/
      for ( int igamma = 0; igamma < sequential_gamma_sets; igamma++ ) {

        char aff_tag_prefix[200];
        sprintf ( aff_tag_prefix, "/N-qbGqqbGq-N/T%d_X%d_Y%d_Z%d/QX%d_QY%d_QZ%d/Gc_%s",
              gsx[0], gsx[1], gsx[2], gsx[3], momentum[0], momentum[1], momentum[2], sequential_gamma_tag[igamma] );


        /***************************************************************************
         * add sequential fht vertex
         ***************************************************************************/
        exitstatus = prepare_sequential_fht_loop_source ( sequential_source[0], loop, propagator_up, sequential_gamma_list[igamma], sequential_gamma_num[igamma], ephase[imom], 1 );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[njjn_fht_invert_contract] Error from prepare_sequential_fht_loop_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(123);
        }

        /***************************************************************************
         * invert the Dirac operator on the sequential source
         *
         * ONLY SINK smearing here
         ***************************************************************************/
        exitstatus = prepare_propagator_from_source ( sequential_propagator, sequential_source[1], 12, _OP_ID_UP, 0, 1, gauge_field_smeared,
            check_propagator_residual, gauge_field_with_phase, lmzz, NULL );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[njjn_fht_invert_contract] Error from prepare_propagator_from_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(123);
        }

        /***************************************************************************
         *
         * p - ubar u - p
         *
         ***************************************************************************/

        /***************************************************************************
         * fill the fermion propagator fp with sink-smeared propagator_up
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp, propagator_up_snk_smeared, VOLUME);

        /***************************************************************************
         * fill the fermion propagator fp2 with sequential_propagator
         *   is already sink-smeared
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp2, sequential_propagator, VOLUME);

        /***************************************************************************
         * fill the fermion propagator fp3 with sink-smeared propagator_dn
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp3, propagator_dn_snk_smeared, VOLUME);

        /***************************************************************************
         * contractions as for N-N diagrams n1, n2,
         * but with sequential up - after - up in two different places
         ***************************************************************************/
        for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
        for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {
    
          /***************************************************************************
           * here we calculate fp3 = Gamma[if2] x propagator_dn / fp3 x Gamma[if1]
           ***************************************************************************/
          fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp3, VOLUME );
    
          fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );
    
          fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );
    
          /***************************************************************************
           * diagram t1
           ***************************************************************************/
          sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t1", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
    
          /* the actual contraction
           *   the operation is called type v5 here  */
          exitstatus = contract_v5 ( vx, fp2, fp3, fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v5, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
          exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          /* write to AFF file */
          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
    
          /***************************************************************************
           * diagram t2
           ***************************************************************************/
          sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t2", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
    
          exitstatus = contract_v6 ( vx, fp2, fp3, fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v6, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /***************************************************************************
           * diagram t3
           ***************************************************************************/
          sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t3", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
    
          /* the actual contraction
           *   the operation is called type v5 here  */
          exitstatus = contract_v5 ( vx, fp, fp3, fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v5, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
          exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          /* write to AFF file */
          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
    
          /***************************************************************************
           * diagram t4
           ***************************************************************************/
          sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t4", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

          exitstatus = contract_v6 ( vx, fp, fp3, fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v6, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
        }} // end of loop on Dirac gamma structures

        /***************************************************************************/
        /***************************************************************************/

        /***************************************************************************
         *
         * n - ubar u - n
         *
         ***************************************************************************/

        /***************************************************************************
         * fill the fermion propagator fp with sink-smeared propagator_up
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp, propagator_up_snk_smeared, VOLUME);

        /***************************************************************************
         * fill the fermion propagator fp2 with sink-smeared propagator_dn
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp2, propagator_dn_snk_smeared, VOLUME);

        /***************************************************************************
         * fill the fermion propagator fp3 with sequential_propagator
         *   is already sink-smeared
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp3, sequential_propagator, VOLUME);

        /***************************************************************************
         * contractions as for N-N diagrams n1, n2,
         * but with sequential up - after - up in two different places
         ***************************************************************************/
        for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
        for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {
    
          /***************************************************************************
           * here we calculate fp3 = Gamma[if2] x propagator_dn / fp3 x Gamma[if1]
           ***************************************************************************/
          fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp3, VOLUME );
    
          fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );
    
          fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );
    
          /***************************************************************************
           * diagram t7
           ***************************************************************************/
          sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t7", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
    
          /* the actual contraction
           *   the operation is called type v5 here  */
          exitstatus = contract_v5 ( vx, fp2, fp3, fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v5, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
          exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          /* write to AFF file */
          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
    
          /***************************************************************************
           * diagram t8
           ***************************************************************************/
          sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t8", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

          exitstatus = contract_v6 ( vx, fp2, fp3, fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v6, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

        }} // end of loop on Dirac gamma structures

        /***************************************************************************/
        /***************************************************************************/

      } // end of loop on sequential source gamma matrices

      /***************************************************************************/
      /***************************************************************************/

      /***************************************************************************
       ***************************************************************************
       **
       ** Part III sequential down - after - down inversion and contraction
       **
       ***************************************************************************
       ***************************************************************************/

      /***************************************************************************
       * multiply the Fourier phase
       *
       * sequential source = exp( i p_seq x ) propagator_dn
       * for each spin color component isc and
       * for each timeslice it
       ***************************************************************************/
      for ( int isc = 0; isc < 12; isc++ ) {
        for ( int it = 0; it < T; it++ ) {
          /* memory offset for timeslice it */
          size_t const offset =  it * _GSI(VOL3);
          spinor_field_eq_spinor_field_ti_complex_field ( sequential_source[0][isc] + offset, propagator_dn[isc] + offset, (double*)(ephase[imom]), VOL3 );
        }
      }

      /***************************************************************************
       * loop on sequential source gamma matrices
       ***************************************************************************/
      for ( int igamma = 0; igamma < g_sequential_source_gamma_id_number; igamma++ ) {

        int gamma_id = g_sequential_source_gamma_id_list[igamma];

        /***************************************************************************
         * multiply the sequential gamma matrix
         ***************************************************************************/
        for ( int isc = 0; isc < 12; isc++ ) {
          spinor_field_eq_gamma_ti_spinor_field ( sequential_source[1][isc], gamma_id, sequential_source[0][isc], VOLUME );
        }

        /***************************************************************************
         * invert the Dirac operator on the sequential source
         *
         * ONLY SINK smearing here
         ***************************************************************************/
        exitstatus = prepare_propagator_from_source ( sequential_propagator, sequential_source[1], 12, _OP_ID_DN, 0, 1, gauge_field_smeared, check_propagator_residual, gauge_field_with_phase, lmzz, NULL );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[njjn_fht_invert_contract] Error from prepare_propagator_from_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(123);
        }

        /***************************************************************************
         *
         * p - dbar d - p
         *
         ***************************************************************************/

        /***************************************************************************
         * fill the fermion propagator fp with sink-smeared propagator_up
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp, propagator_up_snk_smeared, VOLUME);

        /***************************************************************************
         * fill the fermion propagator fp2 with sequential_propagator
         *   is sink-smeared
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp2, sequential_propagator, VOLUME);

        /***************************************************************************
         * contractions for N-J-N diagrams t1, t2 with sequential down - after - down
         ***************************************************************************/
        for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
        for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {

          /***************************************************************************
           * here we calculate fp3 = Gamma[if2] x sequential propagator / fp2 x Gamma[if1]
           ***************************************************************************/
          fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp2, VOLUME );
    
          fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );
    
          fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );
    
          /***************************************************************************
           * diagram t5
           ***************************************************************************/
          sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t5", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

          /* the actual contraction
           *   the operation is called type v5 here  */
          exitstatus = contract_v5 ( vx, fp, fp3, fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v5, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
          exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          /* write to AFF file */
          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
    
          /***************************************************************************
           * diagram t6
           ***************************************************************************/
          sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t6", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

          exitstatus = contract_v6 ( vx, fp, fp3, fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v6, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
    
        }}  /* end of loop on Dirac Gamma structures */

        /***************************************************************************/
        /***************************************************************************/
 
        /***************************************************************************
         *
         * n - dbar d - n
         *
         ***************************************************************************/

        /***************************************************************************
         * fill the fermion propagator fp  with sink-smeared propagator_up
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp, propagator_up_snk_smeared, VOLUME);

        /***************************************************************************
         * fill the fermion propagator fp2 with sink-smeared propagator_dn
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp2, propagator_dn_snk_smeared, VOLUME);

        /***************************************************************************
         * fill the fermion propagator fp3 with sequential_propagator
         *   is sink-smeared
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp3, sequential_propagator, VOLUME);

        /***************************************************************************
         * contractions for N-J-N diagrams t1, t2 with sequential down - after - down
         ***************************************************************************/
        for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
        for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {

          /***************************************************************************
           * here we calculate fp = Gamma[if2] x propagator_up / fp x Gamma[if1]
           ***************************************************************************/
          fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp, gamma_f1_list[if2], fp, VOLUME );
    
          fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp, gamma_f1_list[if1], fp, VOLUME );
    
          fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp, fp, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );
    
          /***************************************************************************
           * diagram t9
           ***************************************************************************/
          sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t9", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

          /* the actual contraction
           *   the operation is called type v5 here  */
          exitstatus = contract_v5 ( vx, fp2, fp, fp3, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v5, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
          exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          /* write to AFF file */
          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
    
          /***************************************************************************
           * diagram t10
           ***************************************************************************/
          sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t10", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

          exitstatus = contract_v6 ( vx, fp2, fp, fp3, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v6, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
    
          /***************************************************************************
           * diagram t11
           ***************************************************************************/
          sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t11", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

          /* the actual contraction
           *   the operation is called type v5 here  */
          exitstatus = contract_v5 ( vx, fp3, fp, fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v5, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
          exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          /* write to AFF file */
          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
    
          /***************************************************************************
           * diagram t12
           ***************************************************************************/
          sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t12", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

          exitstatus = contract_v6 ( vx, fp3, fp, fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_v6, status was %d\n", exitstatus);
            EXIT(48);
          }
    
          exitstatus = contract_vn_momentum_projection ( vp, vx, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

        }}  /* end of loop on Dirac Gamma structures */

        /***************************************************************************/
        /***************************************************************************/


      }  /* end of loop on sequential source gamma ids */

      fini_3level_dtable ( &sequential_source );

    }  /* end of loop on sequential source momenta */

    /***************************************************************************
     * clean up
     ***************************************************************************/
    free_fp_field ( &fp  );
    free_fp_field ( &fp2 );
    free_fp_field ( &fp3 );
    fini_2level_dtable ( &vx );
    fini_3level_dtable ( &vp );

#ifdef HAVE_LHPC_AFF
    /***************************************************************************
     * I/O process id 2 closes its AFF writer
     ***************************************************************************/
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

    /***************************************************************************
     * free propagator fields
     ***************************************************************************/
    fini_2level_dtable ( &propagator_up );
    fini_2level_dtable ( &propagator_dn );
    fini_2level_dtable ( &propagator_up_snk_smeared );
    fini_2level_dtable ( &propagator_dn_snk_smeared );
    fini_2level_dtable ( &sequential_propagator );

  }  /* end of loop on source locations */

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/
  fini_2level_ztable ( &ephase );

  fini_rng_state ( &g_rng_state);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  if ( gauge_field_with_phase != NULL ) free ( gauge_field_with_phase );
  if ( gauge_field_smeared    != NULL ) free ( gauge_field_smeared );

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

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [njjn_fht_invert_contract] %s# [njjn_fht_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [njjn_fht_invert_contract] %s# [njjn_fht_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}