/****************************************************
 * p2gg_exdefl_analyse
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
#include "scalar_products.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "clover.h"

int get_momentum_id ( int const p[3], int const q[3], int const n, int (* const r)[3] ) {

  int const s[3] = {
    -( p[0] + q[0] ),
    -( p[1] + q[1] ),
    -( p[2] + q[2] ) };

  for ( int i = 0; i < n; i++ ) {
    if ( r[i][0] == s[0] && r[i][1] == s[1] && r[i][2] == s[2] ) return ( i );
  }
  return( -1 );
}  /* end of get_momentum_id */


using namespace cvc;

int main(int argc, char **argv) {

  const char infile_prefix[] = "p2gg";
  const char outfile_prefix[] = "p2gg_exdefl_analyse";

  int c;
  int filename_set = 0;
  /* int check_position_space_WI=0; */
  int exitstatus;
  char filename[100];
  int evecs_num = 0;
  struct timeval ta, tb, start_time, end_time;
  char key[400];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'n':
      evecs_num = atoi ( optarg );
      break;
    case 'h':
    case '?':
    default:
      exit(1);
      break;
    }
  }

  gettimeofday ( &start_time, (struct timezone *)NULL );


  /* set the default values */
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [p2gg_exdefl_analyse] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [p2gg_exdefl_analyse] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_exdefl_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_exdefl_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_exdefl_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_exdefl_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  int io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[p2gg_exdefl_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [p2gg_exdefl_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

#ifdef HAVE_LHPC_AFF
  /***********************************************************
   * writer for aff output file
   ***********************************************************/
  struct AffReader_s *affr = NULL;
  if(io_proc == 2) {
    sprintf ( filename, "%s.%.4d.nev%d.aff", infile_prefix, Nconf, evecs_num );
    fprintf(stdout, "# [p2gg_exdefl_analyse] reading data from file %s\n", filename);
    affr = aff_reader ( filename );
    const char * aff_status_str = aff_reader_errstr ( affr );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[p2gg_exdefl_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  }  /* end of if io_proc == 2 */
#endif

  /***********************************************************
   *
   * read the eigenvalues
   *
   ***********************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  /***********************************************************
   * allocate and read
   ***********************************************************/
  double * evecs_eval = init_1level_dtable ( evecs_num );
  if ( evecs_eval == NULL ) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(11);
  }

  double * evecs_eval_inv = init_1level_dtable ( evecs_num );
  if ( evecs_eval_inv == NULL ) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(11);
  }

  struct AffNode_s * affn = aff_reader_root( affr );
  if( affn == NULL ) {
    fprintf(stderr, "[p2gg_exdefl_analyse] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
    EXIT(17);
  }

  /* AFF read */
  uint32_t uitems = ( uint32_t )evecs_num;

  sprintf ( key, "/eval/C%d/N%d", Nconf, evecs_num );
  if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_exdefl_analyse] reading key %s %s %d\n", key, __FILE__, __LINE__ );

  struct AffNode_s * affdir = aff_reader_chpath ( affr, affn, key );
  if ( affdir == NULL ) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_writer_mkpath %s %d\n", __FILE__, __LINE__);
    EXIT(15);
  }

  exitstatus = aff_node_get_double ( affr, affdir, evecs_eval, uitems );
  if(exitstatus != 0) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_node_get_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(16);
  }

  for ( int iw = 0; iw < evecs_num; iw++ ) {
    evecs_eval_inv[iw] = 1. / evecs_eval[iw];
  }

  if ( g_verbose > 2 ) {
    for ( int iw = 0; iw < evecs_num; iw++ ) {
      fprintf( stdout, "# [] eval %6d   %25.16e   %25.16e\n", iw, evecs_eval[iw] , evecs_eval_inv[iw] );
    }
  }

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "p2gg_exdefl_analyse", "aff-read-evecs", g_cart_id == 0 );

  /***********************************************************
   *
   * read p mat
   *
   ***********************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  /***********************************************************
   * allocate
   ***********************************************************/
  double _Complex *** vw_mat_p = init_3level_ztable ( T_global, evecs_num, evecs_num );
  if ( vw_mat_p == NULL ) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(11);
  }
  
  int source_momentum[3] = {0,0,0};
  int source_gamma_id = 4;

  /* AFF read */
  uitems = ( uint32_t )T_global * evecs_num * evecs_num;

  sprintf ( key, "/vdag-gp-w/C%d/N%d/PX%d_PY%d_PZ%d/G%d", Nconf, evecs_num,
      source_momentum[0], source_momentum[1], source_momentum[2], source_gamma_id);
  if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_exdefl_analyse] reading key %s %s %d\n", key, __FILE__, __LINE__ );

  affdir = aff_reader_chpath ( affr, affn, key );
  if ( affdir == NULL ) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
    EXIT(15);
  }

  exitstatus = aff_node_get_complex ( affr, affdir, vw_mat_p[0][0], uitems );
  if(exitstatus != 0) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(16);
  }

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "p2gg_exdefl_analyse", "aff-read-mat-p", g_cart_id == 0 );

  /***********************************************************
   *
   * read v mat
   *
   ***********************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  /***********************************************************
   * allocate
   ***********************************************************/
  double _Complex ***** vw_mat_v = init_5level_ztable ( g_sink_momentum_number, g_sink_gamma_id_number, T_global, evecs_num, evecs_num );
  if ( vw_mat_v == NULL ) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(11);
  }
  

  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
    int sink_momentum[3] = {
      g_sink_momentum_list[imom][0],
      g_sink_momentum_list[imom][1],
      g_sink_momentum_list[imom][2] };

    for ( int igam = 0; igam < g_sink_gamma_id_number; igam++ ) {
      int sink_gamma_id = g_sink_gamma_id_list[igam];

      /* AFF read */
      uitems = ( uint32_t )T_global * evecs_num * evecs_num;

      sprintf ( key, "/vdag-gp-w/C%d/N%d/PX%d_PY%d_PZ%d/G%d", Nconf, evecs_num,
          sink_momentum[0], sink_momentum[1], sink_momentum[2], sink_gamma_id);
      if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_exdefl_analyse] reading key %s %s %d\n", key, __FILE__, __LINE__ );

      affdir = aff_reader_chpath ( affr, affn, key );
      if ( affdir == NULL ) {
        fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
        EXIT(15);
      }

      exitstatus = aff_node_get_complex ( affr, affdir, vw_mat_v[imom][igam][0][0], uitems );
      if(exitstatus != 0) {
        fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(16);
      }
    }
  }

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "p2gg_exdefl_analyse", "aff-read-mat-p", g_cart_id == 0 );

  /***********************************************************
   * loop on upper limit of eigenvectors
   ***********************************************************/
  int evecs_use_step = 1;
  int evecs_use_min  = 1;
  for ( int evecs_use = evecs_use_min; evecs_use <= evecs_num; evecs_use += evecs_use_step ) {

    double _Complex * loop_p = init_1level_ztable ( T_global );
    if ( loop_p == NULL ) {
      fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_1level_ztable %s %d\n", __FILE__, __LINE__);
      EXIT(15);
    }

    /***********************************************************
     * partial trace
     ***********************************************************/
#pragma omp parallel for
    for ( int x0 = 0; x0 < T_global; x0++ ) {
      for ( int iw = 0; iw < evecs_use; iw++ ) {
        loop_p[x0] += vw_mat_p[x0][iw][iw] * evecs_eval_inv[iw];
      }
    }

    /***********************************************************
     * write loop to file
     ***********************************************************/
    sprintf ( filename, "%s.loop.g%d.px%d_py%d_pz%d.nev%d.%.4d", outfile_prefix, source_gamma_id,
        source_momentum[0], source_momentum[1], source_momentum[2], evecs_use, Nconf ); 
    FILE * ofs = fopen ( filename, "w" );
    if ( ofs == NULL ) {
      fprintf ( stderr, "[p2gg_exdefl_analyse] Error from fopen %s %d\n", __FILE__, __LINE__);
      EXIT(21);
    }

    for ( int x0 = 0; x0 < T_global; x0++ ) {
      fprintf ( ofs, "%25.16e  %25.16e\n", creal(loop_p[x0]), cimag(loop_p[x0]) );
    }

    fclose ( ofs );

    double _Complex ***** corr_v = init_5level_ztable ( g_sink_momentum_number, g_sink_gamma_id_number, g_sink_gamma_id_number, T_global, T_global );
    if ( corr_v == NULL ) {
      fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__);
      EXIT(15);
    }
    
    double _Complex ** corr_3pt = init_2level_ztable ( g_sequential_source_timeslice_number, T_global );
    if ( corr_3pt == NULL ) {
      fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
      EXIT(15);
    }

    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

      int kmom = get_momentum_id ( g_sink_momentum_list[imom], source_momentum, g_sink_momentum_number, g_sink_momentum_list );

      if ( kmom == -1 ) continue;
      if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_exdefl_analyse] psrc = %3d %3d %3d   psnk = %3d %3d %3d   pcur = %3d %3d %3d\n",
          source_momentum[0], source_momentum[1], source_momentum[2],
          g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
          g_sink_momentum_list[kmom][0], g_sink_momentum_list[kmom][1], g_sink_momentum_list[kmom][2] );

      for ( int ig1 = 0; ig1 < g_sink_gamma_id_number; ig1++ ) {
      for ( int ig2 = 0; ig2 < g_sink_gamma_id_number; ig2++ ) {

#pragma omp parallel for
        for ( int x0 = 0; x0 < T_global; x0++ ) {
        for ( int y0 = 0; y0 < T_global; y0++ ) {
          for ( int iw = 0; iw < evecs_use; iw++ ) {
          for ( int iv = 0; iv < evecs_use; iv++ ) {
            corr_v[imom][ig1][ig2][x0][y0] += vw_mat_v[imom][ig1][x0][iw][iv] * vw_mat_v[kmom][ig2][y0][iv][iw] * evecs_eval_inv[iw] * evecs_eval_inv[iv];
          }}
        }}
      }}
    }

    int const epsilon_tensor[3][3] = { {0,1,2}, {1,2,0}, {2,0,1} };

    double pvec[3] = {
        2 * sin ( M_PI * g_sink_momentum_list[0][0] / (double)LX_global ),
        2 * sin ( M_PI * g_sink_momentum_list[0][1] / (double)LY_global ),
        2 * sin ( M_PI * g_sink_momentum_list[0][2] / (double)LZ_global ) };

    double const norm = 1. / ( pvec[0] * pvec[0] + pvec[1] * pvec[1] + pvec[2] * pvec[2] ) / (double)g_sink_momentum_number;

    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

      pvec[0] = 2 * sin ( M_PI * g_sink_momentum_list[imom][0] / (double)LX_global );
      pvec[1] = 2 * sin ( M_PI * g_sink_momentum_list[imom][1] / (double)LY_global );
      pvec[2] = 2 * sin ( M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global );

      for ( int iperm = 0; iperm < 3; iperm++ ) {
        int const ia = epsilon_tensor[iperm][0];
        int const ib = epsilon_tensor[iperm][1];
        int const ic = epsilon_tensor[iperm][2];

        for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {

#pragma omp parallel for
          for ( int tsrc = 0; tsrc < T_global; tsrc++ ) {
            int tsnk = ( tsrc + g_sequential_source_timeslice_list[idt] + T_global ) % T_global;

            for ( int tcur = 0; tcur < T_global; tcur++ ) {
              int itsc = ( tcur - tsrc + T_global ) % T_global;

              corr_3pt[idt][itsc] += 
                  creal( loop_p[tsrc] ) *  creal( corr_v[imom][ia][ib][tcur][tsnk] ) * pvec[ic] * norm 
                + cimag( loop_p[tsrc] ) *  cimag( corr_v[imom][ia][ib][tcur][tsnk] ) * pvec[ic] * norm *I;
            }
          }
        }
      }
    }

    /***********************************************************
     * write loop to file
     ***********************************************************/
    sprintf ( filename, "%s.3pt.disc.g%d.px%d_py%d_pz%d.qx%d_qy%d_qz%d.nev%d.%.4d", outfile_prefix, source_gamma_id,
        source_momentum[0], source_momentum[1], source_momentum[2], 
        g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
        evecs_use, Nconf ); 
    ofs = fopen ( filename, "w" );
    if ( ofs == NULL ) {
      fprintf ( stderr, "[p2gg_exdefl_analyse] Error from fopen %s %d\n", __FILE__, __LINE__);
      EXIT(21);
    }

    for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {
      for ( int itsc = 0; itsc < T_global; itsc++ ) {
        fprintf ( ofs, "%3d %3d %25.16e  %25.16e\n", 
            g_sequential_source_timeslice_list[idt], itsc, creal( corr_3pt[idt][itsc]), cimag( corr_3pt[idt][itsc]) );
      }
    }

    fclose ( ofs );

    fini_1level_ztable ( &loop_p );
    fini_5level_ztable ( &corr_v );
    fini_2level_ztable ( &corr_3pt );

  }  /* end of loop on upper limits */

  fini_5level_ztable ( &vw_mat_v );
  fini_3level_ztable ( &vw_mat_p );

  /***********************************************************
   * close writer
   ***********************************************************/
#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    aff_reader_close ( affr );
  }  /* end of if io_proc == 2 */
#endif
  
  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/
  
  fini_1level_dtable ( &evecs_eval );
  fini_1level_dtable ( &evecs_eval_inv );

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "p2gg_exdefl_analyse", "total time", g_cart_id == 0 );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_exdefl_analyse] end of run\n");
    fprintf(stderr, "# [p2gg_exdefl_analyse] end of run\n");
  }

  return(0);
}