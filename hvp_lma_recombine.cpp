/****************************************************
 * hvp_lma_recombine.c
 *
 * Do 29. Mär 16:01:03 CEST 2018
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

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

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "table_init_d.h"
#include "table_init_z.h"
#include "clover.h"
#include "gsp_utils.h"
#include "gsp_recombine.h"


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform cvc correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -w                  : check position space WI [default false]\n");
  EXIT(0);
}

int get_momentum_id ( int const p[3], int (* const p_list)[3], int const n ) {

  for ( int i = 0; i < n; i++ ) {
    if ( ( p[0] == p_list[i][0] ) &&
         ( p[1] == p_list[i][1] ) &&
         ( p[2] == p_list[i][2] ) ) {
      return(i);
    }
  }

  return(-1);
}  // end of get_momentum_id



int main(int argc, char **argv) {
  
  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[100];
  // double ratime, retime;
  unsigned int evecs_block_length = 0;
  unsigned int evecs_num = 0;
  unsigned int nproc_t = 1;


#ifdef HAVE_LHPC_AFF
  char tag[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "sh?f:b:n:t:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'b':
      evecs_block_length = atoi ( optarg );
      break;
    case 'n':
      evecs_num = atoi( optarg );
      break;
    case 't':
      nproc_t = atoi( optarg );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  g_the_time = time(NULL);

  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "p2gg.input");
  // fprintf(stdout, "# [hvp_lma_recombine] Reading input from file %s\n", filename);
  read_input_parser(filename);


  /***********************************************************
   * initialize MPI parameters for cvc
   ***********************************************************/
  mpi_init(argc, argv);

  /***********************************************************
   * set number of openmp threads
   ***********************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hvp_lma_recombine] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hvp_lma_recombine] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hvp_lma_recombine] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * initialize geometry fields
   ***********************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0) {
    fprintf(stderr, "[hvp_lma_recombine] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_contraction(2);
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * read eigenvalues
   ***********************************************************/
  double * evecs_eval = NULL;
  sprintf( tag, "/hvp/eval/N%d", evecs_num );
  sprintf( filename, "%s.%.4d.eval", filename_prefix, Nconf );
  exitstatus = gsp_read_eval( &evecs_eval, evecs_num, filename, tag);
  if( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma_recombine] Error from gsp_read_eval, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(39);
  }


  /***********************************************************
   * set auxilliary eigenvalue fields
   ***********************************************************/
  double * const evecs_lambdainv           = (double*)malloc(evecs_num*sizeof(double));
  double * const evecs_4kappasqr_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if( evecs_lambdainv == NULL || evecs_4kappasqr_lambdainv == NULL ) {
    fprintf(stderr, "[hvp_lma_recombine] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }
  for( unsigned int i = 0; i < evecs_num; i++) {
    evecs_lambdainv[i]           = 2.* g_kappa / evecs_eval[i];
    evecs_4kappasqr_lambdainv[i] = 4.* g_kappa * g_kappa / evecs_eval[i];
  }

  /***********************************************************
   * check evecs_block_length
   ***********************************************************/
  if ( evecs_block_length == 0 ) {
    evecs_block_length = evecs_num;
    if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma_recombine] WARNING, reset evecs_block_length to %u\n", evecs_num );
  }

  /***********************************************************
   * set io process
   ***********************************************************/
  int const io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[hvp_lma_recombine] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [hvp_lma_recombine] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***********************************************************/
  /***********************************************************/

  sprintf( tag, "/hvp/lma/N%d/B%d", evecs_num, evecs_block_length );
  sprintf( filename, "%s.%.4d", filename_prefix, Nconf );

  double _Complex ***** phi = init_5level_ztable ( 4, g_sink_momentum_number, T, evecs_num, evecs_num );
  if ( phi == NULL ) {
    fprintf (stderr, "[hvp_lma_recombine] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(3);
  }

  for ( int imu = 0; imu < 4; imu++ ) {

    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

      // loop on timeslices
      for ( int it = 0; it < T; it++ ) {
      
        exitstatus = gsp_read_cvc_node ( phi[imu][imom][it], evecs_num, evecs_block_length, g_sink_momentum_list[imom], "mu", imu, filename, tag, it, nproc_t );

        if( exitstatus != 0 ) {
          fprintf(stderr, "[hvp_lma_recombine] Error from gsp_read_cvc_node, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(14);
        }

        // TEST
        // write the complete Nev x Nev complex field to stdout
        if ( g_verbose > 4 )  {
          // show the data read by gsp_read_cvc_node
          fprintf ( stdout, "# [hvp_lma_recombine] /hvp/lma/N%d/t%.2d/mu%d/px%.2dpy%.2dpz%.2d\n", evecs_num, it, imu,
              g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] );
          for ( unsigned int i1 = 0; i1 < evecs_num; i1++ ) {
          for ( unsigned int i2 = 0; i2 < evecs_num; i2++ ) {
            fprintf ( stdout, "  %4d  %4d    %25.16e   %25.16e\n", i1, i2, 
                creal( phi[imu][imom][it][i1][i2] ), cimag( phi[imu][imom][it][i1][i2] ) );
          }}
        }  // end of if verbose > 4

      }  // end of loop on timeslices

      /***********************************************************/
      /***********************************************************/

#if 0
    /***********************************************************
     * test loops
     ***********************************************************/
      double _Complex *phi_tr = init_1level_ztable ( T );

      gsp_tr_mat_weight ( phi_tr , phi[imu][imom] , evecs_4kappasqr_lambdainv , evecs_num, T );

      if ( g_verbose > 4 )  {
        // show the trace
        fprintf ( stdout, "# [hvp_lma_recombine] /loop/cvc/nev%.4d/px%.2dpy%.2dpz%.2d/mu%d\n", evecs_num, g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], imu );
        for ( int it = 0; it < T; it++ ) {
          fprintf ( stdout, "%26.16e  %25.16e\n", creal( phi_tr[it] ), cimag( phi_tr[it] ) );
        }
      }  // end of if verbose > 4

      fini_1level_ztable ( &phi_tr );
#endif  // of if 0

    }  // end of loop on momenta

  }  // end of loop on mu

  /***********************************************************/
  /***********************************************************/

#if 0
  /***********************************************************
   * test Ward identity
   ***********************************************************/

  if ( g_verbose > 4 )  {
    // show the trace

    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

      int psource[3] = {
        -g_sink_momentum_list[imom][0],
        -g_sink_momentum_list[imom][1],
        -g_sink_momentum_list[imom][2] };
      int imom2 = get_momentum_id ( psource, g_sink_momentum_list, g_sink_momentum_number );
      if ( imom2 == -1 ) {
        fprintf( stderr, "[hvp_lma_recombine] Error, could not find matching id for psource = %3d %3d %3d\n", 
            psource[0], psource[1], psource[2] );
        
        EXIT(4);
      }
      if ( g_verbose > 4 ) {
        fprintf ( stdout, "# [hvp_lma_recombine] sink momentum %3d %3d %3d  source momentum %3d %3d %3d\n",
            g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] ,
            g_sink_momentum_list[imom2][0], g_sink_momentum_list[imom2][1], g_sink_momentum_list[imom2][2] );
      }

      for ( int imu = 0; imu < 4; imu++ ) {
      for ( int inu = 0; inu < 4; inu++ ) {

        double _Complex *phi_tr = init_1level_ztable ( T );

        gsp_tr_mat_weight_mat_weight ( phi_tr, phi[imu][imom], phi[inu][imom2], evecs_4kappasqr_lambdainv, evecs_num, T );

        fprintf ( stdout, "# [hvp_lma_recombine] /hvp/cvc/nev%.4d/px%.2dpy%.2dpz%.2d/mu%d/nu%d\n", evecs_num, g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], imu , inu );
        for ( int it = 0; it < T; it++ ) {
          fprintf ( stdout, "%26.16e  %25.16e\n", creal( phi_tr[it] ), cimag( phi_tr[it] ) );
        }

        fini_1level_ztable ( &phi_tr );

      }}
    }  // end of loop on momenta
  }  // end of if verbose > 4
#endif


  /***********************************************************/
  /***********************************************************/

  fini_5level_ztable ( &phi );

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/

  if ( evecs_lambdainv           != NULL ) free ( evecs_lambdainv );
  if ( evecs_4kappasqr_lambdainv != NULL ) free ( evecs_4kappasqr_lambdainv ); 
  if ( evecs_eval                != NULL ) free ( evecs_eval );

  /***********************************************************
   * free clover matrix terms
   ***********************************************************/
  fini_clover ();

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_xchange_eo_propagator();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif


  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [hvp_lma_recombine] %s# [hvp_lma_recombine] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [hvp_lma_recombine] %s# [hvp_lma_recombine] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
