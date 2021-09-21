/****************************************************
 * avgx_invert_contract
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
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
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "contract_cvc_tensor.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "Q_phi.h"
#include "clover.h"
#include "ranlxd.h"
#include "smearing_techniques.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1
#define _OP_ID_ST 2

#define _DERIV  1
#define _DDERIV 0

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to calculate charged pion FF inversions + contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual   [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "cpff";

  const char fbwd_str[2][4] =  { "fwd", "bwd" };
  
  const char flavor_tag[4][2] =  { "u", "d", "sp", "sm" };

  int const current_momentum_sqr_max = 3;

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  size_t sizeof_spinor_field;
  char filename[400];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double **smzz[2] = { NULL, NULL }, **smzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL, *gauge_field_smeared = NULL;
  char output_filename[400];
  int * rng_state = NULL;
  int spin_dilution = 4;
  int color_dilution = 1;


  int const gamma_current_number = 10;
  int gamma_current_list[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

  int const gamma_v_number = 4;
  int gamma_v_list[4] = { 0, 1, 2, 3  };

  char data_tag[400];
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  struct AffWriter_s *affw = NULL;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "rh?f:s:c:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'r':
      check_propagator_residual = 1;
      break;
    case 's':
      spin_dilution = atoi ( optarg );
      break;
    case 'c':
      color_dilution = atoi ( optarg );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) sprintf ( filename, "%s.input", outfile_prefix );
  /* fprintf(stdout, "# [avgx_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [avgx_invert_contract] calling tmLQCD wrapper init functions\n");

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
  /* exitstatus = tmLQCD_invert_init(argc, argv, 1); */
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

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [avgx_invert_contract] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[avgx_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);

  /***************************************************************************
   * some additional xchange operations
   ***************************************************************************/
  mpi_init_xchange_contraction(2);
  mpi_init_xchange_eo_spinor ();

  /***************************************************************************
   * initialize own gauge field or get from tmLQCD wrapper
   ***************************************************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [avgx_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [avgx_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[avgx_invert_contract] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[avgx_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[avgx_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif


  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[avgx_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[avgx_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv for light quark
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[avgx_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv for strange quark
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &smzz, &smzzinv, gauge_field_with_phase, g_mus, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[avgx_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
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
        fprintf(stderr, "[avgx_invert_contract] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
    }  /* end of if N_aoe > 0 */
  }  /* end of if N_Jacobi > 0 */



  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[avgx_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [avgx_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***********************************************************
   * set operator ids depending on fermion type
   ***********************************************************/
#if 0
  if ( g_fermion_type == _TM_FERMION ) {
    op_id_up = 0;
    op_id_dn = 1;
  } else if ( g_fermion_type == _WILSON_FERMION ) {
    op_id_up = 0;
    op_id_dn = 0;
  }
#endif

  /***************************************************************************
   * allocate memory for spinor fields 
   * WITH HALO
   ***************************************************************************/
  size_t nelem = _GSI( VOLUME+RAND );
  double ** spinor_work  = init_2level_dtable ( 2, nelem );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[avgx_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * allocate memory for spinor fields
   * WITHOUT halo
   ***************************************************************************/
  int const spin_color_dilution = spin_dilution * color_dilution;
  nelem = _GSI( VOLUME );
  double *** stochastic_propagator_mom_list = init_3level_dtable ( 2, spin_color_dilution, nelem );
  if ( stochastic_propagator_mom_list == NULL ) {
    fprintf(stderr, "[avgx_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  /***************************************************************************/

  double *** stochastic_propagator_zero_list = init_2level_dtable ( 4, spin_color_dilution, nelem );
  if ( stochastic_propagator_zero_list == NULL ) {
    fprintf(stderr, "[avgx_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double *** stochastic_propagator_zero_smeared_list = init_2level_dtable ( 4, spin_color_dilution, nelem );
  if ( stochastic_propagator_zero_smeared_list == NULL ) {
    fprintf(stderr, "[avgx_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  /***************************************************************************/

  double ** stochastic_source_list = init_2level_dtable ( spin_color_dilution, nelem );
  if ( stochastic_source_list == NULL ) {
    fprintf(stderr, "[avgx_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }


  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  exitstatus = init_rng_state ( g_seed, &rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[avgx_invert_contract] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }
 
  if ( g_verbose > 4 ) {
    for ( int i = 0; i < rlxd_size(); i++ ) {
      fprintf ( stdout, "rng %2d %10d\n", g_cart_id, rng_state[i] );
    }
  }

  /***************************************************************************
   * loop on source timeslices
   ***************************************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

    /***************************************************************************
     * random source timeslice
     ***************************************************************************/
    double dts;
    ranlxd ( &dts , 1 );
    int gts = (int)(dts * T_global);

    /***************************************************************************
     * local source timeslice and source process ids
     ***************************************************************************/

    int source_timeslice = -1;
    int source_proc_id   = -1;

    exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[avgx_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

#if ( defined HAVE_LHPC_AFF ) && !(defined HAVE_HDF5 )
    /***************************************************************************
     * output filename
     ***************************************************************************/
    sprintf ( output_filename, "%s.%.4d.t%d.aff", outfile_prefix, Nconf, gts );
    /***************************************************************************
     * writer for aff output file
     ***************************************************************************/
    if(io_proc == 2) {
      affw = aff_writer ( output_filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[avgx_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#elif ( defined HAVE_HDF5 )
    sprintf ( output_filename, "%s.%.4d.t%d.h5", outfile_prefix, Nconf, gts );
#endif
    if(io_proc == 2 && g_verbose > 1 ) { 
      fprintf(stdout, "# [avgx_invert_contract] writing data to file %s\n", output_filename);
    }

    /***************************************************************************
     * re-initialize random number generator
     ***************************************************************************/
    /*
    if ( ! g_read_source ) {
      sprintf(filename, "rng_stat.%.4d.tsrc%.3d.stochastic-oet.out", Nconf, gts );
      exitstatus = init_rng_stat_file ( ( ( gts + 1 ) * 10000 + g_seed ), filename );
      if(exitstatus != 0) {
        fprintf(stderr, "[avgx_invert_contract] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }
    }
    */

    /***************************************************************************
     * loop on stochastic oet samples
     ***************************************************************************/
    for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

      /***************************************************************************
       * synchronize rng states to state at zero
       ***************************************************************************/
      exitstatus = sync_rng_state ( rng_state, 0, 0 );
      if(exitstatus != 0) {
        fprintf(stderr, "[avgx_invert_contract] Error from sync_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }

      /***************************************************************************
       * read stochastic oet source from file
       ***************************************************************************/
      if ( g_read_source ) {
        for ( int i = 0; i < spin_color_dilution; i++ ) {
          sprintf(filename, "%s.%.4d.t%d.%d.%.5d", filename_prefix, Nconf, gts, i, isample);
          if ( ( exitstatus = read_lime_spinor( stochastic_source_list[i], filename, 0) ) != 0 ) {
            fprintf(stderr, "[avgx_invert_contract] Error from read_lime_spinor, status was %d\n", exitstatus);
            EXIT(2);
          }
        }
        /* recover the ran field */
        exitstatus = init_timeslice_source_oet(stochastic_source_list, gts, NULL, spin_dilution, color_dilution,  -1 );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[avgx_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(64);
        }

      /***************************************************************************
       * generate stochastic oet source
       ***************************************************************************/
      } else {
        /* call to initialize the ran field 
         *   penultimate argument is momentum vector for the source, NULL here
         *   final argument in arg list is 1
         */
        if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gts, NULL, spin_dilution, color_dilution, 1 ) ) != 0 ) {
          fprintf(stderr, "[avgx_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(64);
        }
        if ( g_write_source ) {
          for ( int i = 0; i < spin_color_dilution; i++ ) {
            sprintf(filename, "%s.%.4d.t%d.%d.%.5d", filename_prefix, Nconf, gts, i, isample);
            if ( ( exitstatus = write_propagator( stochastic_source_list[i], filename, 0, g_propagator_precision) ) != 0 ) {
              fprintf(stderr, "[avgx_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }
          }
        }
      }  /* end of if read stochastic source - else */

      /***************************************************************************
       * retrieve current rng state and 0 writes his state
       ***************************************************************************/
      exitstatus = get_rng_state ( rng_state );
      if(exitstatus != 0) {
        fprintf(stderr, "[avgx_invert_contract] Error from get_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }

      exitstatus = save_rng_state ( 0, NULL );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[avgx_invert_contract] Error from save_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
        EXIT(38);
      }

          
      /***************************************************************************
       * prepare stochastic timeslice source at source momentum
       ***************************************************************************/

      int source_momentum[3] = { 0, 0, 0 };
      exitstatus = init_timeslice_source_oet ( stochastic_source_list, gts, source_momentum, spin_dilution, color_dilution, 0 );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[avgx_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(64);
      }

      if ( N_Jacobi > 0 ) {
        /***************************************************************************
         * SOURCE SMEARING
         ***************************************************************************/
        for( int i = 0; i < spin_color_dilution; i++) {
          exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_source_list[i], N_Jacobi, kappa_Jacobi);
          if(exitstatus != 0) {
            fprintf(stderr, "[avgx_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            return(11);
          }
        }
      }

      /***************************************************************************
       * loop on quark flavor
       ***************************************************************************/
      for ( int iflavor = 0; iflavor < 4; iflavor++ ) {

        /***************************************************************************
         * invert for stochastic timeslice propagator at zero momentum
         *   dn flavor
         *   this one will run from source to sink as part of the sequential
         *   propagator
         ***************************************************************************/
        for( int i = 0; i < spin_color_dilution; i++) {
  
          memcpy ( spinor_work[0], stochastic_source_list[i], sizeof_spinor_field );
  
  
          memset ( spinor_work[1], 0, sizeof_spinor_field );
  
          exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
          if(exitstatus < 0) {
            fprintf(stderr, "[avgx_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(44);
          }
  
          if ( check_propagator_residual ) {
            if ( iflavor < 2 ) {
              check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
            } else {
              check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, smzz[iflavor%2], smzzinv[iflavor%2], 1 );
            }
          }
  
          memcpy( stochastic_propagator_zero_list[iflavor][i], spinor_work[1], sizeof_spinor_field);

          memcpy( stochastic_propagator_zero_smeared_list[iflavor][i], spinor_work[1], sizeof_spinor_field);
 
          if ( N_Jacobi > 0 ) {
            /***************************************************************************
             * SOURCE SMEARING
             ***************************************************************************/
            exitstatus = Jacobi_Smearing ( gauge_field_smeared,  stochastic_propagator_zero_smeared_list[iflavor][i], N_Jacobi, kappa_Jacobi);
            if(exitstatus != 0) {
              fprintf(stderr, "[avgx_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              return(11);
            }
          }

        }  /* end of loop on spin color dilution indices */
  
  
        /***************************************************************************
         * invert for stochastic timeslice propagator at source momenta
         ***************************************************************************/
        for ( int isrc_mom = 0; isrc_mom < g_source_momentum_number; isrc_mom++ ) {
  
          /***************************************************************************
           * NOTE: we take the negative of the momentum in the list
           * since we use it in the daggered timeslice propagator
           ***************************************************************************/
          int source_momentum[3] = {
              g_source_momentum_list[isrc_mom][0],
              g_source_momentum_list[isrc_mom][1],
              g_source_momentum_list[isrc_mom][2] };
  
          /***************************************************************************
           * prepare stochastic timeslice source at source momentum
           ***************************************************************************/
          exitstatus = init_timeslice_source_oet ( stochastic_source_list, gts, source_momentum, spin_dilution, color_dilution, 0 );
          if( exitstatus != 0 ) {
            fprintf(stderr, "[avgx_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(64);
          }

          /***************************************************************************
           * source-smearing
           ***************************************************************************/
          if ( N_Jacobi > 0 ) {
            for( int i = 0; i < spin_color_dilution; i++) {
              exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_source_list[i], N_Jacobi, kappa_Jacobi);
              if(exitstatus != 0) {
                fprintf(stderr, "[avgx_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                return(11);
              }
            }
          }

          /***************************************************************************
           * invert
           ***************************************************************************/
          for ( int iflavor = 0; iflavor < 2; iflavor++ ) {
          
            for( int i = 0; i < spin_color_dilution; i++) {

              memcpy ( spinor_work[0], stochastic_source_list[i], sizeof_spinor_field );
   
              memset ( spinor_work[1], 0, sizeof_spinor_field );
  
              exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
              if(exitstatus < 0) {
                fprintf(stderr, "[avgx_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(44);
              }
  
              if ( check_propagator_residual ) {
                check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
              }
  
              memcpy( stochastic_propagator_mom_list[iflavor][i], spinor_work[1], sizeof_spinor_field);
  
            }  /* end of loop on spinor components */

          }  /* end of loop on flavor */

          double *** stochastic_propagator_mom_smeared_list = init_3level_dtable ( 2, spin_color_dilution, _GSI(VOLUME) );
          if ( stochastic_propagator_mom_smeared_list == NULL ) {
            fprintf(stderr, "[avgx_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(48);
          }

          /***************************************************************************
           * SINK SMEARING of the momentum propagator
           ***************************************************************************/
          memcpy ( stochastic_propagator_mom_smeared_list[0][0], stochastic_propagator_mom_list[0][0], 2 * spin_color_dilution * _GSI(VOLUME) );

          if ( N_Jacobi > 0 ) {
 
            for ( int iflavor = 0; iflavor < 2; iflavor++ ) {
              for( int i = 0; i < spin_color_dilution; i++) {

                exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_propagator_mom_smeared_list[iflavor][i], N_Jacobi, kappa_Jacobi);
                if(exitstatus != 0) {
                  fprintf(stderr, "[avgx_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  return(11);
                }
              }
            }
          }
 
          /***************************************************************************
           * loop on quark flavors
           *
           * for Tr [ X(0) Gf Y(p) Gi ] = Xbar(0)^+ g5 Gf Y(p) Gi g5
           * X = u,d,s+,s-, at zero momentum
           * Y = u,d, at non-zero momentum
           ***************************************************************************/
          for ( int iflavor = 0; iflavor < 4; iflavor++ ) {
          
            for ( int iflavor2 = 0; iflavor2 < 2; iflavor2++ ) {

              /***************************************************************************
               * contractions for 2-point functions
               ***************************************************************************/
              for ( int isrc_gamma = 0; isrc_gamma < g_source_gamma_id_number; isrc_gamma++ )
              {
              
                /* for ( int isnk_gamma = 0; isnk_gamma < g_sink_gamma_id_number;   isnk_gamma++ ) 
                { */

                int isink_gamma = isrc_gamma;
          
                  /* allocate contraction fields in position and momentum space */
                  double * contr_x = init_1level_dtable ( 2 * VOLUME );
                  if ( contr_x == NULL ) {
                    fprintf(stderr, "[avgx_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
                    EXIT(3);
                  }
    
                  double ** contr_p = init_2level_dtable ( g_sink_momentum_number , 2 * T );
                  if ( contr_p == NULL ) {
                    fprintf(stderr, "[avgx_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
                    EXIT(3);
                  }
    
                  /* contractions in x-space */
                  contract_twopoint_xdep ( contr_x, g_source_gamma_id_list[isrc_gamma], g_sink_gamma_id_list[isnk_gamma], 
                      stochastic_propagator_zero_smeared_list[2*(iflavor/2) + 1-(iflavor%2)], 
                      stochastic_propagator_mom_smeared_list[iflavor2],
                      spin_dilution, color_dilution, 1, 1., 64 );
    
                  int sink_momentum[3] = {
                    -source_momentum[0],
                    -source_momentum[1],
                    -source_momentum[2] };
  
                  /* momentum projection at sink */
                  exitstatus = momentum_projection ( contr_x, contr_p[0], T, 1, &sink_momentum );
                  if(exitstatus != 0) {
                    fprintf(stderr, "[avgx_invert_contract] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(3);
                  }
    
                  sprintf ( data_tag, "/%s-gf-%s-gi/mus%6.4f/t%d/s%d/gf%d/gi%d/pix%dpiy%dpiz%d", flavor_tag[iflavor], flavor_tag[iflavor2],
                      g_mus,
                      gts, isample,
                      g_source_gamma_id_list[isnk_gamma], g_source_gamma_id_list[isrc_gamma],
                      source_momentum[0], source_momentum[1], source_momentum[2] );
    
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
                  exitstatus = contract_write_to_aff_file ( contr_p, affw, data_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
#elif ( defined HAVE_HDF5 )          
                  exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
#endif
                  if(exitstatus != 0) {
                    fprintf(stderr, "[avgx_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    return(3);
                  }
     
                  /* deallocate the contraction fields */       
                  fini_1level_dtable ( &contr_x );
                  fini_2level_dtable ( &contr_p );
  
                /* } */  /* end of loop on gamma at sink */
              }  /* end of loop on gammas at source */

            }  /* end of loop on flavor at sink */
          }  /* end of loop on flavor at source */
  
          /***************************************************************************/

          /*****************************************************************
           *
           * sequential
           *
           *****************************************************************/
  
 
           double ** sequential_propagator_list = init_2level_dtable ( spin_color_dilution, nelem );
           if ( sequential_propagator_list == NULL ) {
             fprintf(stderr, "[avgx_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );;
             EXIT(48);
           }

           int sink_momentum[3] = {
               -source_momentum[0],
               -source_momentum[1],
               -source_momentum[2] };

          /*****************************************************************
           * seq source mom is zero
           *****************************************************************/
          int seq_source_momentum[3] = { 0, 0, 0 };
  
          for ( int iflavor = 0; iflavor < 2; iflavor++ ) {
          
            for ( int iflavor2 = 0; iflavor2 < 2; iflavor2++ ) {

              /*****************************************************************
               * sequential source gamma id is 
               * 4 = id for same flavor, i.e. sbar_+ g5 u and sbar_- g5 d
               * 5 = g5 for opposite flavor, i.e. sbar_+ g5 d and sbar_- g5 u
               *****************************************************************/
              seq_source_gamma = ( iflavor == iflavor2 ) ? 4 : 5;
  
              /*****************************************************************
               * loop on sequential source timeslices
               *****************************************************************/
              for ( int iseq_timeslice = 0; iseq_timeslice < g_sequential_source_timeslice_number; iseq_timeslice++ ) {
  
                /*****************************************************************
                 * global sequential source timeslice
                 * NOTE: counted from current source timeslice
                 *****************************************************************/
                int gtseq = ( gts + g_sequential_source_timeslice_list[iseq_timeslice] + T_global ) % T_global;
  
                /*****************************************************************
                 * invert for sequential timeslice propagator
                 *****************************************************************/
                for ( int i = 0; i < spin_color_dilution; i++ ) {
  
                  /*****************************************************************
                   * prepare sequential timeslice source 
                   *
                   * THROUGH THE SINK, so use the SINK SMEARED stochastic zero momentum propagator
                   *****************************************************************/
       
                  /*****************************************************************
                   * STRANGE-AFTER-LIGHT
                   *****************************************************************/
                  exitstatus = init_sequential_source ( spinor_work[0], stochastic_propagator_mom_smeared_list[iflavor][i], gtseq, seq_source_momentum, seq_source_gamma );
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[avgx_invert_contract] Error from init_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(64);
                  }
  
                  if ( N_Jacobi > 0 ) {
                    /***************************************************************************
                     * SINK SMEARING THE SEQUENTIAL SOURCE
                     ***************************************************************************/
                    exitstatus = Jacobi_Smearing ( gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi);
                    if(exitstatus != 0) {
                      fprintf(stderr, "[avgx_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      return(11);
                    }
                  }
 
                  memset ( spinor_work[1], 0, sizeof_spinor_field );
  
                  exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], 2+iflavor2 );
                  if(exitstatus < 0) {
                    fprintf(stderr, "[avgx_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                    EXIT(44);
                  }
  
                  if ( check_propagator_residual ) {
                    check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, smzz[iflavor2], smzzinv[iflavor2], 1 );
                  }
  
                  /***************************************************************************
                   * NO SMEARING AT THIS END OF THE PRPOAGATOR
                   *
                   * this end runs to the insertion
                   ***************************************************************************/
  
                  memcpy( sequential_propagator_list[i], spinor_work[1], sizeof_spinor_field );
                }  /* end of loop on oet spin components */
  
                /*****************************************************************/
                /*****************************************************************/
  
                /*****************************************************************
                 * contractions for local current insertion
                 *
                 * stochastic_propagator_mom_list is dn type
                 *
                 * sequential_propagator_list is strange-after-light type
                 * (sequential through the sink
                 *
                 *                 /\
                 *                /  \
                 * s^+ = sbar    /    \ s
                 *              /      \
                 *             /________\
                 *                 l
                 *****************************************************************/
  
                /*****************************************************************/
                /*****************************************************************/
  
                /*****************************************************************
                 * contractions for covariant displacement insertion
                 *****************************************************************/
  
                /*****************************************************************
                 * loop on fbwd for covariant displacement
                 *****************************************************************/
                for ( int ifbwd = 0; ifbwd <= 1; ifbwd++ ) {
  
                  double ** sequential_propagator_displ_list     = init_2level_dtable ( spin_color_dilution, _GSI(VOLUME) );
                  if ( sequential_propagator_displ_list == NULL ) {
                    fprintf(stderr, "[avgx_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
                    EXIT(33);
                  }
                
                  /*****************************************************************
                   * loop on directions for covariant displacement
                   *****************************************************************/
                  for ( int mu = 0; mu < 4; mu++ ) {
  
                    for ( int i = 0; i < spin_color_dilution; i++ ) {
                      exitstatus = spinor_field_eq_cov_displ_spinor_field ( sequential_propagator_displ_list[i], sequential_propagator_list[i], mu, ifbwd, gauge_field_with_phase );
                      if ( exitstatus != 0 ) {
                        fprintf(stderr, "[avgx_invert_contract] Error from spinor_field_eq_cov_displ_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(33);
                      }
                    }
  

                    for ( int icurrent_gamma = 0; icurrent_gamma < gamma_v_number; icurrent_gamma++ )
	            {
                 
                      int gamma_current = gamma_v_list[icurrent_gamma];

                      int gamma_source = seq_source_gamma;

                      double * contr_p = init_1level_dtable (  2*T );
                      if ( contr_p == NULL ) {
                        fprintf(stderr, "[avgx_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
                        EXIT(47);
                      }
                  
                      contract_twopoint_snk_momentum ( contr_p, gamma_source,  gamma_current, 
                            stochastic_propagator_zero_list[ 2*(iflavor2/2) + 1 - (iflavor2%2) ], 
                            sequential_propagator_displ_list, spin_dilution, color_dilution, current_momentum, 1);
 STOPPED HERE 
  
                      sprintf ( data_tag, "/%s-gd-s%s%s-gi/t%d/s%d/dt%d/gf%d/gc%d/d%d/%s/gi%d/pfx%dpfy%dpfz%d/", 
                          flavor_tag[2+iflavor2],
                          flavor_tag[2+iflavor2], flavor_tag[iflavor],
                          gts, isample, g_sequential_source_timeslice_list[iseq_timeslice],
                          seq_source_gamma, gamma_current, mu, fbwd_str[ifbwd], gamma_source,
                          seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2] );
  
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
                      exitstatus = contract_write_to_aff_file ( contr_p, affw, data_tag, g_source_momentum_list, g_source_momentum_number, io_proc );
#elif ( defined HAVE_HDF5 )
                      exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, g_source_momentum_list, g_source_momentum_number, io_proc );
#endif
                      if(exitstatus != 0) {
                        fprintf(stderr, "[avgx_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(3);
                      }
                
                      fini_2level_dtable ( &contr_p );
                      
                    }  /* end of loop on current gamma */
  
                  }  /* end of loop on mu */

                }  /* end of loop on fbwd */

              }  /* loop on sequential source timeslices */

              /*****************************************************************/
              /*****************************************************************/
STOPPED HERE
              /*****************************************************************
               * loop on sequential source timeslices
               *****************************************************************/
              for ( int iseq_timeslice = 0; iseq_timeslice < g_sequential_source_timeslice_number; iseq_timeslice++ ) {
  
                /*****************************************************************
                 * global sequential source timeslice
                 * NOTE: counted from current source timeslice
                 *****************************************************************/
                int gtseq = ( gts + g_sequential_source_timeslice_list[iseq_timeslice] + T_global ) % T_global;
  
                /*****************************************************************
                 * invert for sequential timeslice propagator
                 *****************************************************************/
                for ( int i = 0; i < spin_color_dilution; i++ ) {
  
                  /*****************************************************************
                   * prepare sequential timeslice source 
                   *
                   * THROUGH THE SINK, so use the SINK SMEARED stochastic zero momentum propagator
                   *****************************************************************/
       
                  /*****************************************************************
                   * LIGHT-AFTER-STRANGE
                   *****************************************************************/
                  exitstatus = init_sequential_source ( spinor_work[0], stochastic_propagator_zero_smeared_list[2+iflavor][i], gtseq, seq_source_momentum, seq_source_gamma );
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[avgx_invert_contract] Error from init_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(64);
                  }
  
                  if ( N_Jacobi > 0 ) {
                    /***************************************************************************
                     * SINK SMEARING THE SEQUENTIAL SOURCE
                     ***************************************************************************/
                    exitstatus = Jacobi_Smearing ( gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi);
                    if(exitstatus != 0) {
                      fprintf(stderr, "[avgx_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      return(11);
                    }
                  }
 
                  memset ( spinor_work[1], 0, sizeof_spinor_field );
  
                  exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor2 );
                  if(exitstatus < 0) {
                    fprintf(stderr, "[avgx_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                    EXIT(44);
                  }
  
                  if ( check_propagator_residual ) {
                    check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor2], mzzinv[iflavor2], 1 );
                  }
  
                  /***************************************************************************
                   * NO SMEARING AT THIS END OF THE PRPOAGATOR
                   *
                   * this end runs to the insertion
                   ***************************************************************************/
  
                  memcpy( sequential_propagator_list[i], spinor_work[1], sizeof_spinor_field );
                }  /* end of loop on oet spin components */
  
                /*****************************************************************/
                /*****************************************************************/
  
                /*****************************************************************
                 * contractions for local current insertion
                 *
                 * stochastic_propagator_mom_list is dn type
                 *
                 * sequential_propagator_list is light-after-strange type
                 * (sequential through the sink
                 *
                 *                 /\
                 *                /  \
                 * l^+ = lbar    /    \ l
                 *              /      \
                 *             /________\
                 *                 s
                 *****************************************************************/
  
                /*****************************************************************/
                /*****************************************************************/
  
                /*****************************************************************
                 * contractions for covariant displacement insertion
                 *****************************************************************/
  
                /*****************************************************************
                 * loop on fbwd for covariant displacement
                 *****************************************************************/
                for ( int ifbwd = 0; ifbwd <= 1; ifbwd++ ) {
  
                  double ** sequential_propagator_displ_list     = init_2level_dtable ( spin_color_dilution, _GSI(VOLUME) );
                  if ( sequential_propagator_displ_list == NULL ) {
                    fprintf(stderr, "[avgx_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
                    EXIT(33);
                  }
                
                  /*****************************************************************
                   * loop on directions for covariant displacement
                   *****************************************************************/
                  for ( int mu = 0; mu < 4; mu++ ) {
  
                    for ( int i = 0; i < spin_color_dilution; i++ ) {
                      exitstatus = spinor_field_eq_cov_displ_spinor_field ( sequential_propagator_displ_list[i], sequential_propagator_list[i], mu, ifbwd, gauge_field_with_phase );
                      if ( exitstatus != 0 ) {
                        fprintf(stderr, "[avgx_invert_contract] Error from spinor_field_eq_cov_displ_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(33);
                      }
                    }
  

                    for ( int icurrent_gamma = 0; icurrent_gamma < gamma_v_number; icurrent_gamma++ )
	            {
                 
                      int gamma_current = gamma_v_list[icurrent_gamma];

                      int gamma_source = seq_source_gamma;

                      double * contr_p = init_1level_dtable (  2*T );
                      if ( contr_p == NULL ) {
                        fprintf(stderr, "[avgx_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
                        EXIT(47);
                      }
                  
                      contract_twopoint_snk_momentum ( contr_p, gamma_source,  gamma_current, 
                            stochastic_propagator_mom_list[ 1 - iflavor2 ], 
                            sequential_propagator_displ_list, spin_dilution, color_dilution, current_momentum, 1);
  
  
                      sprintf ( data_tag, "/%s-gd-s%s%s-gi/t%d/s%d/dt%d/gf%d/gc%d/d%d/%s/gi%d/pfx%dpfy%dpfz%d/", 
                          flavor_tag[iflavor2],
                          flavor_tag[iflavor2], flavor_tag[2+iflavor],
                          gts, isample, g_sequential_source_timeslice_list[iseq_timeslice],
                          seq_source_gamma, gamma_current, mu, fbwd_str[ifbwd], gamma_source,
                          seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2] );
  
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
                      exitstatus = contract_write_to_aff_file ( contr_p, affw, data_tag, g_source_momentum_list, g_source_momentum_number, io_proc );
#elif ( defined HAVE_HDF5 )
                      exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, g_source_momentum_list, g_source_momentum_number, io_proc );
#endif
                      if(exitstatus != 0) {
                        fprintf(stderr, "[avgx_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(3);
                      }
                
                      fini_2level_dtable ( &contr_p );
                      
                    }  /* end of loop on current gamma */
  
                  }  /* end of loop on mu */

                }  /* end of loop on fbwd */

              }  /* loop on sequential source timeslices */

              /*****************************************************************/
              /*****************************************************************/
  
            }  /* end of loop on flavor2 */
  
          }  /* end of loop on flavors */

          exitstatus = init_timeslice_source_oet ( NULL, -1, NULL, 0, 0, -2 );

        }  /* end of loop on source momenta */

      }
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
        if(io_proc == 2) {
          const char * aff_status_str = (char*)aff_writer_close (affw);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[avgx_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
          EXIT(32);
        }
      }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

    }  /* end of loop on samoles */

  }  /* end of loop on source locaions */

  /***************************************************************************
   * decallocate spinor fields
   ***************************************************************************/
  fini_3level_dtable ( &stochastic_propagator_mom_list );
  fini_2level_dtable ( &stochastic_propagator_zero_list );
  fini_2level_dtable ( &stochastic_propagator_zero_smeared_list );
  fini_2level_dtable ( &stochastic_source_list );
  fini_2level_dtable ( &sequential_propagator_list );
  fini_2level_dtable ( &spinor_work );

  /***************************************************************************
   * fini rng state
   ***************************************************************************/
  fini_rng_state ( &rng_state);

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  free( gauge_field_with_phase );
  free( gauge_field_smeared );

  /* free clover matrix terms */
  fini_clover ( &mzz, &mzzinv );
  fini_clover ( &smzz, &smzzinv );


#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif

  free_geometry();


#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor ();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [avgx_invert_contract] %s# [avgx_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [avgx_invert_contract] %s# [avgx_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}