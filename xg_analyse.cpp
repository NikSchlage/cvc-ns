/****************************************************
 * xg_analyse 
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

#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "contract_cvc_tensor.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_i.h"
#include "gamma.h"
#include "uwerr.h"
#include "derived_quantities.h"

#ifndef _SQR
#define _SQR(_a) ((_a)*(_a))
#endif

#define _LOOP_ANALYSIS

#undef _XG_PION
#define _XG_NUCLEON
#undef _XG_CHARGED

#define _TWOP_STATS

#define _RAT_METHOD
#undef _FHT_METHOD_ALLT
#undef _FHT_METHOD_ACCUM

#undef _RAT_SUB_METHOD


#define MAX_SMEARING_LEVELS 40

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse < Y-O_g Ybar > correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default cpff.input]\n");
  EXIT(0);
}


/**********************************************************
 *
 **********************************************************/
inline void write_data_real ( double ** data, char * filename, int *** lst, unsigned int const n0, unsigned int const n1 ) {

  FILE * ofs = fopen ( filename, "w" );
  if ( ofs == NULL ) {
    fprintf ( stderr, "[write_data_real] Error from fopen %s %d\n",  __FILE__, __LINE__ );
    EXIT(1);
  }

  for ( unsigned int i0 = 0; i0 < n0; i0++ ) {
    fprintf ( ofs, "# %c %6d\n", lst[i0][0][0], lst[i0][0][1] );
    for ( unsigned int i1 = 0; i1 < n1; i1++ ) {
      fprintf ( ofs, "%25.16e\n", data[i0][i1] );
    }
  }

  fclose ( ofs );
}  /* end of write_data_real */


/**********************************************************
 *
 **********************************************************/
inline void write_data_real2_reim ( double **** data, char * filename, int *** lst, unsigned int const n0, unsigned int const n1, unsigned int const n2, int const ri ) {

  FILE * ofs = fopen ( filename, "w" );
  if ( ofs == NULL ) {
    fprintf ( stderr, "[write_data_real2_reim] Error from fopen %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  for ( unsigned int i0 = 0; i0 < n0; i0++ ) {
  for ( unsigned int i1 = 0; i1 < n1; i1++ ) {
    fprintf ( ofs , "# %c %6d %3d %3d %3d %3d\n", lst[i0][i1][0], lst[i0][i1][1], lst[i0][i1][2], lst[i0][i1][3], lst[i0][i1][4], lst[i0][i1][5] );

    for ( unsigned int i2 = 0; i2 < n2; i2++ ) {
      fprintf ( ofs, "%25.16e\n", data[i0][i1][i2][ri] );
    }
  }}
  fclose ( ofs );
}  /* end of write_data_real2_reim */

/**********************************************************
 *
 **********************************************************/
inline void src_avg_real2_reim ( double ** data, double ****corr, unsigned int const n0, unsigned int const n1, unsigned int const n2, int const ri ) {

#pragma omp parallel for
  for ( unsigned int iconf = 0; iconf < n0; iconf++ ) {
    for ( unsigned int it = 0; it < n2; it++ ) {
      double dtmp = 0.;

      for ( unsigned int isrc = 0; isrc < n1; isrc++ ) {
        dtmp += corr[iconf][isrc][it][ri];
      }
      data[iconf][it] = dtmp / (double)n1;
    }
  }
}  /* end of src_avg_real2_reim */

/**********************************************************
 *
 **********************************************************/
int main(int argc, char **argv) {
  
  const double TWO_MPI = 2. * M_PI;

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



  char const reim_str[2][3] = { "re", "im" };

  char const correlator_prefix[2][20] = { "local-local" , "charged"};

  char const flavor_tag[2][20]        = { "d-gf-u-gi" , "u-gf-d-gi" };

  const char insertion_operator_name[1][20] = { "plaquette" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[600];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "NA";
  int twop_fold_propagator = 0;
  int write_data = 0;
  int operator_type = -1;
  struct timeval ta, tb;
  unsigned int stout_level_iter[MAX_SMEARING_LEVELS];
  double stout_level_rho[MAX_SMEARING_LEVELS];
  unsigned int stout_level_num = 0;
  int insertion_operator_type = 0;
  double temp_spat_weight[2] = { 1., -1. };

 #ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char key[400];
#endif


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:F:R:E:w:O:s:W:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [xg_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [xg_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'F':
      twop_fold_propagator = atoi ( optarg );
      fprintf ( stdout, "# [xg_analyse] twop fold_propagator set to %d\n", twop_fold_propagator );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [xg_analyse] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [xg_analyse] write_date set to %d\n", write_data );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [xg_analyse] operator_type set to %d\n", operator_type );
      break;
    case 's':
      sscanf ( optarg, "%d,%lf", stout_level_iter+stout_level_num, stout_level_rho+stout_level_num );
      fprintf ( stdout, "# [xg_analyse] stout_level %d  iter %2d  rho %6.4f \n", stout_level_num, stout_level_iter[stout_level_num], stout_level_rho[stout_level_num] );
      stout_level_num++;
      break;
    case 'W':
      sscanf( optarg, "%lf,%lf", temp_spat_weight, temp_spat_weight+1 );
      fprintf ( stdout, "# [xg_analyse] temp_spat_weigt set to %25.16e / %25.16e\n", temp_spat_weight[0], temp_spat_weight[1] );
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
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [xg_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [xg_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [xg_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [xg_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[xg_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[xg_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[xg_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [xg_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  /* sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf); */
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[xg_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[xg_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [xg_analyse] comment %s\n", line );
      continue;
    }
    int itmp[5];
    char ctmp;

    sscanf( line, "%c %d %d %d %d %d", &ctmp, itmp, itmp+1, itmp+2, itmp+3, itmp+4 );

    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][0] = (int)ctmp;
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][1] = itmp[0];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][2] = itmp[1];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][3] = itmp[2];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][4] = itmp[3];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][5] = itmp[4];

    count++;
  }

  fclose ( ofs );


  if ( g_verbose > 3 ) {
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        fprintf ( stdout, "conf_src_list %c %6d %3d %3d %3d %3d\n", 
            conf_src_list[iconf][isrc][0],
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5] );
      }
    }
  }

  /***********************************************************
   * loop on gamma at sink
   ***********************************************************/
  for ( int igf = 0; igf < g_sink_gamma_id_number; igf++ ) {

  /***********************************************************
   * loop on gamma at source
   ***********************************************************/
  for ( int igi = 0; igi < g_source_gamma_id_number; igi++ ) {

    /**********************************************************
     **********************************************************
     ** 
     ** READ DATA
     ** 
     **********************************************************
     **********************************************************/

    /***********************************************************
     * read twop function data
     ***********************************************************/
    double ****** twop = NULL;

    twop = init_6level_dtable ( g_sink_momentum_number, num_conf, num_src_per_conf, 2, T_global, 2 );
    if( twop == NULL ) {
      fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT (24);
    }

#ifdef _XG_PION
    /***********************************************************
     * loop on configs
     ***********************************************************/
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
  
      Nconf = conf_src_list[iconf][0][1];
 
      /***********************************************************
       * loop on sources
       ***********************************************************/
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
  
        for ( int iflavor = 0; iflavor <= 1 ; iflavor++ ) {
   
          gettimeofday ( &ta, (struct timezone *)NULL );
  
          /***********************************************************
           * open AFF reader
           ***********************************************************/
          struct AffNode_s *affn = NULL, *affdir = NULL;
    
          sprintf( filename, "%s/stream_%c/%d/%s.%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", filename_prefix,
              conf_src_list[iconf][isrc][0], 
              conf_src_list[iconf][isrc][1], 
              correlator_prefix[operator_type], flavor_tag[iflavor],
              conf_src_list[iconf][isrc][1], 
              conf_src_list[iconf][isrc][2], 
              conf_src_list[iconf][isrc][3], 
              conf_src_list[iconf][isrc][4], 
              conf_src_list[iconf][isrc][5] );
  
          fprintf(stdout, "# [xg_analyse] reading data from file %s\n", filename);
          affr = aff_reader ( filename );
          const char * aff_status_str = aff_reader_errstr ( affr );
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[xg_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(15);
          }
  
          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[xg_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
            return(103);
          }
  
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "xg_analyse", "open-init-aff-reader", g_cart_id == 0 );
  
          double ** buffer = init_2level_dtable ( T_global, 2 );
  
          /***********************************************************
           * loop on sink momenta
           ***********************************************************/
          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {
  
            gettimeofday ( &ta, (struct timezone *)NULL );
  
            sprintf( key, "/%s/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d/gi%.2d/px%dpy%dpz%d",
                correlator_prefix[operator_type], flavor_tag[iflavor],
                conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5],
                g_sink_gamma_id_list[igf], g_source_gamma_id_list[igi],
                ( 1 - 2 * iflavor ) * g_sink_momentum_list[ipf][0], 
                ( 1 - 2 * iflavor ) * g_sink_momentum_list[ipf][1], 
                ( 1 - 2 * iflavor ) * g_sink_momentum_list[ipf][2] );
  
            if ( g_verbose > 2 ) fprintf ( stdout, "# [xg_analyse] key = %s\n", key );
  
            affdir = aff_reader_chpath (affr, affn, key );
            if( affdir == NULL ) {
              fprintf(stderr, "[xg_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
              EXIT(105);
            }
  
            uint32_t uitems = T_global;
            exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)buffer[0], uitems );
            if( exitstatus != 0 ) {
              fprintf(stderr, "[xg_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }
  
            /***********************************************************
             * source phase
             ***********************************************************/
            const double phase = -TWO_MPI * ( 1 - 2 * iflavor ) * (
                  conf_src_list[isrc][iconf][3] * g_sink_momentum_list[ipf][0] / (double)LX_global
                + conf_src_list[isrc][iconf][4] * g_sink_momentum_list[ipf][1] / (double)LY_global
                + conf_src_list[isrc][iconf][5] * g_sink_momentum_list[ipf][2] / (double)LZ_global );
  
            const double ephase[2] = { cos( phase ) , sin( phase ) } ;
  
            /***********************************************************
             * order from source time and add source phase
             ***********************************************************/
#pragma omp parallel for
            for ( int it = 0; it < T_global; it++ ) {
              int const itt = ( conf_src_list[iconf][isrc][2] + it ) % T_global; 
              twop[ipf][iconf][isrc][iflavor][it][0] = buffer[itt][0] * ephase[0] - buffer[itt][1] * ephase[1];
              twop[ipf][iconf][isrc][iflavor][it][1] = buffer[itt][1] * ephase[0] + buffer[itt][0] * ephase[1];
            }
  
            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "xg_analyse", "read-twop-tensor-aff", g_cart_id == 0 );
  
          }  /* end of loop on sink momenta */
  
          fini_2level_dtable( &buffer );
  
          /**********************************************************
           * close the reader
           **********************************************************/
          aff_reader_close ( affr );
  
        }  /* end of loop on flavor */

      }  /* end of loop on source locations */
    }  /* end of loop on configs */
#endif  /* end of _XG_PION */

#ifdef _XG_NUCLEON
#if 0
    sprintf ( filename, "stream_%c/%s", conf_src_list[0][0][0], filename_prefix3 );
    FILE *fs = fopen( filename , "r" );
    if ( fs == NULL ) {
      fprintf ( stderr, "[xg_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
      EXIT(2);
    }
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        char line[400];
        int itmp;
        double dtmp[4];
        fgets ( line, 100, fs);
        fprintf( stdout, "# [xg_analyse] reading line %s\n", line);
        for( int it =0; it < T_global; it++ ) {
          fscanf ( fs, "%d %lf %lf %lf %lf\n", &itmp, dtmp+0, dtmp+1, dtmp+2, dtmp+3 );
              
          twop[0][iconf][isrc][0][it][0] = dtmp[0];
          twop[0][iconf][isrc][0][it][1] = dtmp[1];
          twop[0][iconf][isrc][0][(T_global - it)%T_global][0] = -dtmp[2];
          twop[0][iconf][isrc][0][(T_global - it)%T_global][1] = -dtmp[3];
        }
      }
    }
    fclose ( fs );
#endif  /* of if 0  */

    sprintf ( filename, "%s/twop.nucleon_zeromom.SS.dat", filename_prefix3 );
    FILE *fs = fopen( filename , "r" );
    if ( fs == NULL ) {
      fprintf ( stderr, "[xg_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
      EXIT(2);
    }

    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        for( int it =0; it < T_global; it++ ) {
          fscanf ( fs, "%lf %lf %lf %lf\n", 
              twop[0][iconf][isrc][0][it], twop[0][iconf][isrc][0][it]+1,
              twop[0][iconf][isrc][1][it], twop[0][iconf][isrc][1][it]+1 );
        }
      }
    }

    fclose ( fs );
#endif  /* end of _XG_NUCLEON */

#ifdef _XG_CHARGED
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        const int ipf = 0;
        double dtmp;


        sprintf ( filename, "stream_%c/%s.%.2d.%.4d.gf_%s.gi_%s",
            conf_src_list[iconf][isrc][0], correlator_prefix[operator_type], conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][1],
            gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ] );

        FILE *fs = fopen( filename , "r" );
        if ( fs == NULL ) {
          fprintf ( stderr, "[xg_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT(2);
        } else {
          if ( g_verbose > 2 ) fprintf( stdout, "# [xg_analyse] reading data from file %s %s %d\n", filename, __FILE__, __LINE__ );
        }
        fscanf( fs, "%lf%lf\n", twop[ipf][iconf][isrc][0][0], &dtmp );
        for ( int it = 1; it < T_global/2; it++ ) {
          fscanf( fs, "%lf%lf\n", twop[ipf][iconf][isrc][0][it] , twop[ipf][iconf][isrc][0][T_global - it] );
        }
        fscanf( fs, "%lf%lf\n", twop[ipf][iconf][isrc][0][T_global/2], &dtmp );
        fclose ( ofs );
        memcpy( twop[ipf][iconf][isrc][1][0], twop[ipf][iconf][isrc][0][0], T_global*2*sizeof(double) );
      }
    }
#endif  /* of _XG_CHARGED */

    /**********************************************************
     *
     * average 2-pt over momentum orbit
     *
     **********************************************************/

    double **** twop_orbit = init_4level_dtable ( num_conf, num_src_per_conf, T_global, 2 );
    if( twop_orbit == NULL ) {
      fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT (24);
    }

#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        /* averaging starts here */
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            twop_orbit[iconf][isrc][it][0] += ( twop[imom][iconf][isrc][0][it][0] + twop[imom][iconf][isrc][1][it][0] ) * 0.5;
            twop_orbit[iconf][isrc][it][1] += ( twop[imom][iconf][isrc][0][it][1] + twop[imom][iconf][isrc][1][it][1] ) * 0.5;
          }  /* end of loop on it */
        }  /* end of loop on imom */

        /* multiply norm from averages over momentum orbit and source locations */
        double const norm = 1. / (double)g_sink_momentum_number;
        for ( int it = 0; it < 2*T_global; it++ ) {
          twop_orbit[iconf][isrc][0][it] *= norm;
        }
      }  /* end of loop on isrc */
    }  /* end of loop on iconf */

    /**********************************************************
     * write orbit-averaged data to ascii file, per source
     **********************************************************/

    char obs_name_prefix[200];
#if defined _XG_NUCLEON
    sprintf ( obs_name_prefix, "NN.orbit.gf_%s.gi_%s.px%d_py%d_pz%d",
            gamma_id_to_Cg_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_Cg_ascii[ g_source_gamma_id_list[igi] ],
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2] );
#else
    sprintf ( obs_name_prefix, "%s.%s.orbit.gf_%s.gi_%s.px%d_py%d_pz%d", correlator_prefix[operator_type], flavor_tag[0],
            gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ],
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2] );
#endif

    if ( write_data == 1 ) {
      for ( int ireim = 0; ireim <= 1; ireim++ ) {
        sprintf ( filename, "%s.%s.corr", obs_name_prefix, reim_str[ireim]);

        write_data_real2_reim ( twop_orbit, filename, conf_src_list, num_conf, num_src_per_conf, T_global, ireim );
      }
    }  /* end of if write data */

#ifdef _TWOP_STATS
    /**********************************************************
     * 
     * STATISTICAL ANALYSIS
     * 
     **********************************************************/
    for ( int ireim = 0; ireim < 1; ireim++ ) {

      if ( num_conf < 6 ) {
        fprintf ( stderr, "[xg_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      }

      double ** data = init_2level_dtable ( num_conf, T_global );
      if ( data == NULL ) {
        fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n",  __FILE__, __LINE__ );
        EXIT(1);
      }

      /* fill data array */
      if ( twop_fold_propagator != 0 ) {
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {

          for ( int it = 0; it <= T_global/2; it++ ) {
              int const itt = ( T_global - it ) % T_global;
              data[iconf][it ] = 0.;
              data[iconf][itt] = 0.;
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                data[iconf][it ] += 0.5 * ( twop_orbit[iconf][isrc][it][ireim] + twop_fold_propagator * twop_orbit[iconf][isrc][itt][ireim] );
            } 
            data[iconf][it ] /= (double)num_src_per_conf;
            data[iconf][itt] = data[iconf][it];
           }
        }
      } else {
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {

          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = 0.;
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              data[iconf][it] += twop_orbit[iconf][isrc][it][ireim];
            }
            data[iconf][it] /= (double)num_src_per_conf;
          }
        }
      }

      char obs_name[200];
      sprintf( obs_name, "%s.%s", obs_name_prefix, reim_str[ireim] );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      /**********************************************************
       * write data to ascii file
       **********************************************************/
      if ( write_data == 2 ) {
        sprintf ( filename, "%s.corr", obs_name );
        write_data_real ( data, filename, conf_src_list, num_conf, T_global );
      }  /* end of if write data */

      /**********************************************************
       * acosh ratio for m_eff
       **********************************************************/
      int const Thp1 = T_global / 2 + 1;
      for ( int itau = 1; itau < Thp1/2; itau++ ) {

#ifdef _XG_NUCLEON
        int narg = 2;
        int arg_first[2] = { 0, itau };
        int arg_stride[2] = {1, 1};
        int nT = Thp1 - itau;

        sprintf ( obs_name, "%s.log_ratio.tau%d.%s", obs_name_prefix, itau, reim_str[ireim] );
        exitstatus = apply_uwerr_func ( data[0], num_conf, T_global, nT, narg, arg_first, arg_stride, obs_name, log_ratio_1_1, dlog_ratio_1_1 );
#else
        int narg = 3;
        int arg_first[3] = { 0, 2 * itau, itau };
        int arg_stride[3] = {1,1,1};
        int nT = Thp1 - 2 * itau;
        sprintf ( obs_name, "%s.acosh_ratio.tau%d.%s", obs_name_prefix, itau, reim_str[ireim] );
        exitstatus = apply_uwerr_func ( data[0], num_conf, T_global, nT, narg, arg_first, arg_stride, obs_name, acosh_ratio, dacosh_ratio );
#endif
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(115);
        }
      }  /* end of loop on tau */

      fini_2level_dtable ( &data );
    }  /* end of loop on reim */

#endif  /* of ifdef _TWOP_STATS */


    /**********************************************************
     * loop on stout smearing levels
     **********************************************************/
  
  for ( unsigned int istout = 0; istout < stout_level_num; istout++ ) {

#ifdef _LOOP_ANALYSIS
    /**********************************************************
     *
     * loop fields
     *
     **********************************************************/
    double *** loop = NULL;
  
    loop = init_3level_dtable ( num_conf, T_global, 2 );
    if ( loop == NULL ) {
      fprintf ( stdout, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(25);
    }
  
    /**********************************************************
     *
     * read loop data
     *
     **********************************************************/
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    
      /***********************************************************
       * open AFF reader
       ***********************************************************/
      struct AffNode_s *affn = NULL, *affdir = NULL;
  
      /* sprintf ( filename, "%s/stream_%c/%d/cpff.xg.%d.aff", filename_prefix2, conf_src_list[iconf][0][0], conf_src_list[iconf][0][1], conf_src_list[iconf][0][1] ); */
      /* sprintf ( filename, "stream_%c/cpff.xg.%d.aff", conf_src_list[iconf][0][0], conf_src_list[iconf][0][1], conf_src_list[iconf][0][1] ); */
      sprintf ( filename, "cpff.xg.%d.aff", conf_src_list[iconf][0][1] );
  
      fprintf(stdout, "# [xg_analyse] reading data from file %s\n", filename);
      affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[xg_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
  
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[xg_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }
  
      if ( stout_level_iter[istout] == 0 ) {
        sprintf( key, "/StoutN%d/StoutRho%6.4f/%s", stout_level_iter[istout], stout_level_rho[istout], insertion_operator_name[insertion_operator_type] );
      } else {
        sprintf( key, "/StoutN%d/StoutRho%6.4f", stout_level_iter[istout], stout_level_rho[istout] );
      }
          
      if ( g_verbose > 2 ) fprintf ( stdout, "# [xg_analyse] key = %s\n", key );
  
      affdir = aff_reader_chpath (affr, affn, key );
      if( affdir == NULL ) {
        fprintf(stderr, "[xg_analyse] Error from aff_reader_chpath for key %s %s %d\n", key, __FILE__, __LINE__);
        EXIT(105);
      }
  
      uint32_t uitems = 2 * T_global;
      exitstatus = aff_node_get_double ( affr, affdir, loop[iconf][0], uitems );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[xg_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(105);
      }
  
      aff_reader_close ( affr );

    }  /* end of loop on configs */
  
    char smearing_tag[50];
    sprintf ( smearing_tag, "stout_%d_%6.4f", stout_level_iter[istout], stout_level_rho[istout] );

    /**********************************************************
     * analyse plaquettes
     **********************************************************/
    {

      double ** data = init_2level_dtable ( num_conf, 4 );

#pragma omp parallel for
      for ( int i = 0; i< num_conf; i++ ) {
        data[i][0] = 0.;
        data[i][1] = 0.;
        data[i][2] = 0.;
        data[i][3] = 0.;
        for( int it = 0; it < T_global; it++ ) {
          data[i][0] += loop[i][it][0];
          data[i][1] += loop[i][it][1];
          data[i][2] += loop[i][it][0] + loop[i][it][1];
          data[i][3] += loop[i][it][0] - loop[i][it][1];
        }
        data[i][0] /= (18. * VOLUME);
        data[i][1] /= (18. * VOLUME);
        data[i][2] /= (18. * VOLUME);
        data[i][3] /= (18. * VOLUME);
      }

      char obs_name[100];
      sprintf ( obs_name, "plaquette.%s" , smearing_tag );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[0], num_conf, 4, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      fini_2level_dtable ( &data );

    }  /* end of block */

    /**********************************************************
     *
     * build trace-subtracted tensor
     *
     **********************************************************/
    double ** loop_sub = init_2level_dtable ( num_conf, T_global );
    if ( loop_sub == NULL ) {
      fprintf ( stdout, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(25);
    }
  
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int it = 0; it < T_global; it++ ) {
        loop_sub[iconf][it] = temp_spat_weight[0] * loop[iconf][it][0] + temp_spat_weight[1] * loop[iconf][it][1];
      }
    }  /* end of loop on configs */
  
    fini_3level_dtable ( &loop );
  
    /**********************************************************
     * tag to characterize the loops w.r.t. low-mode and
     * stochastic part
     **********************************************************/
    /**********************************************************
     * write loop_sub to separate ascii file
     **********************************************************/
  
    if ( write_data ) {
      sprintf ( filename, "loop_sub.%s.corr", smearing_tag );
  
      FILE * loop_sub_fs = fopen( filename, "w" );
      if ( loop_sub_fs == NULL ) {
        fprintf ( stderr, "[xg_analyse] Error from fopen %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      } 
  
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        fprintf ( loop_sub_fs , "# %c %6d\n", conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
        for ( int it = 0; it < T_global; it++ ) {
          fprintf ( loop_sub_fs , "%25.16e\n", loop_sub[iconf][it] );
        }
      }
      fclose ( loop_sub_fs );
    }  /* end of if write data */
  
    /**********************************************************
     *
     * STATISTICAL ANALYSIS OF LOOP VEC
     *
     **********************************************************/
  
    if ( num_conf < 6 ) {
      fprintf ( stderr, "[xg_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }
  
    char obs_name[100];
    sprintf ( obs_name, "loop_sub.%s" , smearing_tag );
  
    /* apply UWerr analysis */
    exitstatus = apply_uwerr_real ( loop_sub[0], num_conf, T_global, 0, 1, obs_name );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }
  
#ifdef _RAT_METHOD
    /**********************************************************
     *
     * STATISTICAL ANALYSIS for products and ratios
     *
     * fixed source - sink separation
     *
     **********************************************************/
  
      /**********************************************************
       * loop on source - sink time separations
       **********************************************************/
      for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {
    
        double **** threep_44 = init_4level_dtable ( num_conf, num_src_per_conf, T_global, 2 ) ;
        if ( threep_44 == NULL ) {
          fprintf ( stderr, "[xg_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }
    
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
    
            /* sink time = source time + dt  */
            /* int const tsink  = (  g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][2] + T_global ) % T_global; */
            int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
            /* sink time with time reversal = source time - dt  */
            /* int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][2] + T_global ) % T_global; */
            int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
    
            /* if ( g_verbose > 4 ) fprintf ( stdout, "# [xg_analyse] t_src %3d   dt %3d   tsink %3d tsink2 %3d\n", conf_src_list[iconf][isrc][2],
                g_sequential_source_timeslice_list[idt], tsink, tsink2 ); */
    
            /**********************************************************
             * !!! LOOK OUT:
             *       This includes the momentum orbit average !!!
             **********************************************************/
            for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
    
#if 0
              double const mom[3] = { 
                  2 * M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
                  2 * M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
                  2 * M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global };
#endif
              /* twop values 
               * a = meson 1 at +mom
               */
              double const a_fwd[2] = { twop[imom][iconf][isrc][0][tsink ][0], twop[imom][iconf][isrc][0][tsink ][1] };
              double const a_bwd[2] = { twop[imom][iconf][isrc][0][tsink2][0], twop[imom][iconf][isrc][0][tsink2][1] };

              double const b_fwd[2] = { twop[imom][iconf][isrc][1][tsink ][0], twop[imom][iconf][isrc][1][tsink ][1] };
              double const b_bwd[2] = { twop[imom][iconf][isrc][1][tsink2][0], twop[imom][iconf][isrc][1][tsink2][1] };
    
              /* loop on insertion times */
              for ( int it = 0; it < T_global; it++ ) {
    
                /* fwd 1 insertion time = source time      + it */
                int const tins_fwd_1 = (  it + conf_src_list[iconf][isrc][2]                                           + T_global ) % T_global;
    
                /* fwd 2 insertion time = source time + dt - it */
                int const tins_fwd_2 = ( -it + conf_src_list[iconf][isrc][2] + g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
    
                /* bwd 1 insertion time = source time      - it */
                int const tins_bwd_1 = ( -it + conf_src_list[iconf][isrc][2]                                           + T_global ) % T_global;
    
                /* bwd 2 insertion time = source time - dt + it */
                int const tins_bwd_2 = (  it + conf_src_list[iconf][isrc][2] - g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
    
                if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_average] insertion times tsrc %3d    dt %3d    tc %3d    tins %3d %3d %3d %3d\n",
                    conf_src_list[iconf][isrc][2], g_sequential_source_timeslice_list[idt], it,
                    tins_fwd_1, tins_fwd_2, tins_bwd_1, tins_bwd_2 );
    
                /**********************************************************
                 * O44, real parts only
                 **********************************************************/
#ifdef _XG_NUCLEON
                threep_44[iconf][isrc][it][0] += ( 
                      /* a_fwd[0] * loop_sub[iconf][tins_fwd_1]*/
                    + b_fwd[0] * loop_sub[iconf][tins_bwd_1]
                  ) * 0.5;
#else
#if 0
                threep_44[iconf][isrc][it][0] += ( 
                        ( a_fwd[0] + b_fwd[0] ) * ( loop_sub[iconf][tins_fwd_1] + loop_sub[iconf][tins_fwd_2] ) 
                      + ( a_bwd[0] + b_bwd[0] ) * ( loop_sub[iconf][tins_bwd_1] + loop_sub[iconf][tins_bwd_2] )
                    ) * 0.125;


                threep_44[iconf][isrc][it][0] += ( 
                        ( a_fwd[0] + b_fwd[0] ) * ( loop_sub[iconf][tins_fwd_1] + loop_sub[iconf][tins_fwd_2] ) 
                    ) * 0.25;
#endif   

                threep_44[iconf][isrc][it][0] += ( 
                      a_fwd[0] * ( loop_sub[iconf][tins_fwd_1] + loop_sub[iconf][tins_fwd_2] )
                    + a_bwd[0] * ( loop_sub[iconf][tins_bwd_1] + loop_sub[iconf][tins_bwd_2] )
                    ) * 0.25;
#endif
              }  /* end of loop on it */
    
            }  /* end of loop on imom */
    
            /**********************************************************
             * normalize
             **********************************************************/
            /* O44 simple orbit average */
            double const norm44 = 1. / g_sink_momentum_number;
            for ( int it = 0; it < 2 * T_global; it++ ) {
              threep_44[iconf][isrc][0][it] *= norm44;
            }
          }  /* end of loop on isrc */
        }  /* end of loop on iconf */
    
        /**********************************************************
         * write 3pt function to ascii file, per source
         **********************************************************/
        if ( write_data == 1) {
          /**********************************************************
           * write 44 3pt
           **********************************************************/
          for ( int ireim = 0; ireim < 1; ireim++ ) {
    
#ifdef _XG_NUCLEON
            sprintf ( filename, "threep.gf_%s.gi_%s.g4_D4.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s.corr",
                gamma_id_to_Cg_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_Cg_ascii[ g_source_gamma_id_list[igi] ],
                smearing_tag,
                g_sequential_source_timeslice_list[idt],
                g_sink_momentum_list[0][0],
                g_sink_momentum_list[0][1],
                g_sink_momentum_list[0][2], reim_str[ireim] );
#else
            sprintf ( filename, "threep.gf_%s.gi_%s.g4_D4.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s.corr",
                gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ],
                smearing_tag,
                g_sequential_source_timeslice_list[idt],
                g_sink_momentum_list[0][0],
                g_sink_momentum_list[0][1],
                g_sink_momentum_list[0][2], reim_str[ireim] );
#endif
            write_data_real2_reim ( threep_44, filename, conf_src_list, num_conf, num_src_per_conf, T_global, ireim );
    
          }  /* end of loop on ireim */
    
        }  /* end of if write_data */
    
        /**********************************************************
         *
         * STATISTICAL ANALYSIS for threep
         *
         * with fixed source - sink separation
         *
         **********************************************************/
        for ( int ireim = 0; ireim < 1; ireim++ ) {
    
          if ( num_conf < 6 ) {
            fprintf ( stderr, "[xg_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }
    
          /* double *** data = init_3level_dtable ( num_conf, num_src_per_conf, T_global ); */
          double ** data = init_2level_dtable ( num_conf, T_global );
          if ( data == NULL ) {
            fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }
    
          /**********************************************************
           * threep_44
           **********************************************************/
#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    
            /* for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              data[iconf][isrc][it] = threep_44[iconf][isrc][it][ireim];
            }} */
    
            for ( int it = 0; it < T_global; it++ ) {
              double dtmp = 0.;
              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                dtmp += threep_44[iconf][isrc][it][ireim];
              }
              data[iconf][it] = dtmp / (double)num_src_per_conf;
            }
          }
    
          char obs_name[100];
#ifdef _XG_NUCLEON
          sprintf ( obs_name, "threep.gf_%s.gi_%s.g4_D4.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s",
              gamma_id_to_Cg_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_Cg_ascii[ g_source_gamma_id_list[igi] ],
              smearing_tag,
              g_sequential_source_timeslice_list[idt],
              g_sink_momentum_list[0][0],
              g_sink_momentum_list[0][1],
              g_sink_momentum_list[0][2], reim_str[ireim] );
#else
          sprintf ( obs_name, "threep.gf_%s.gi_%s.g4_D4.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s",
              gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ],
              smearing_tag,
              g_sequential_source_timeslice_list[idt],
              g_sink_momentum_list[0][0],
              g_sink_momentum_list[0][1],
              g_sink_momentum_list[0][2], reim_str[ireim] );
#endif

          /* apply UWerr analysis */
          exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
    
          if ( write_data == 2 ) {
            sprintf ( filename, "%s.corr", obs_name );
            write_data_real ( data, filename, conf_src_list, num_conf, T_global );
          }
    
          fini_2level_dtable ( &data );
        }  /* end of loop on reim */
    
        /**********************************************************
         *
         * STATISTICAL ANALYSIS for ratio 
         *   with source - sink fixed
         *
         **********************************************************/
        for ( int ireim = 0; ireim < 1; ireim++ ) {
    
          /* UWerr parameters */
          int nT = g_sequential_source_timeslice_list[idt] + 1;
          int narg          = 2;
          int arg_first[2]  = { 0, nT };
          int arg_stride[2] = { 1,  0 };
          char obs_name[100];
    
          double ** data = init_2level_dtable ( num_conf, nT + 1 );
          if ( data == NULL ) {
            fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }
    
          /**********************************************************
           * O44
           **********************************************************/
          src_avg_real2_reim ( data, threep_44, num_conf, num_src_per_conf, nT, ireim );
    
#pragma omp parallel for

          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            double dtmp = 0.;
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              /* int const tsink  = (  g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][2] + T_global ) % T_global; */
              int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
#ifdef _XG_NUCLEON
              dtmp += twop_orbit[iconf][isrc][tsink][ireim];
#else
              /* int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][2] + T_global ) % T_global; */
              int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
              dtmp += 0.5 * ( twop_orbit[iconf][isrc][tsink][ireim] + twop_orbit[iconf][isrc][tsink2][ireim] );
#endif
            }
            data[iconf][nT] = dtmp / (double)num_src_per_conf;
          }

#if 0
#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
              for( int it = 0; it < nT; it++ ) {
                data[iconf*num_src_per_conf + isrc][it] = threep_44[iconf][isrc][it][ireim] /  twop_orbit[iconf][isrc][tsink][ireim];
              }
              data[iconf*num_src_per_conf + isrc][nT] = 1.;
            }
          }
#endif
    
#ifdef _XG_NUCLEON
          sprintf ( obs_name, "ratio.gf_%s.gi_%s.g4_D4.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s",
            gamma_id_to_Cg_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_Cg_ascii[ g_source_gamma_id_list[igi] ],
            smearing_tag,
            g_sequential_source_timeslice_list[idt],
            g_sink_momentum_list[0][0],
            g_sink_momentum_list[0][1],
            g_sink_momentum_list[0][2], reim_str[ireim] );
#else
          sprintf ( obs_name, "ratio.gf_%s.gi_%s.g4_D4.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s",
            gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ],
            smearing_tag,
            g_sequential_source_timeslice_list[idt],
            g_sink_momentum_list[0][0],
            g_sink_momentum_list[0][1],
            g_sink_momentum_list[0][2], reim_str[ireim] );
#endif

          exitstatus = apply_uwerr_func ( data[0], num_conf, nT+1, nT, narg, arg_first, arg_stride, obs_name, ratio_1_1, dratio_1_1 );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(115);
          }
    
        }  /* end of loop on reim */
    
#ifdef _RAT_SUB_METHOD

    /**********************************************************
     *
     * STATISTICAL ANALYSIS for ratio 
     *
     * with fixed source - sink separation  and
     *
     * with vev subtraction
     *
     **********************************************************/
  
        for ( int ireim = 0; ireim < 1; ireim++ ) {
    
          if ( num_conf < 6 ) {
            fprintf ( stderr, "[xg_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }
    
          double ** data = init_2level_dtable ( num_conf, T_global + 2 );
          if ( data == NULL ) {
            fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }
    
          /**********************************************************
           * threep_44 sub
           **********************************************************/
#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {

            for ( int it = 0; it < T_global; it++ ) {
              double dtmp = 0.;
              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                dtmp += threep_44[iconf][isrc][it][ireim];
              }
              data[iconf][it] = dtmp / (double)num_src_per_conf;
            }


            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              /* COUNT FROM SOURCE 0 */
              int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
              int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
              data[iconf][T_global] += 0.5 * ( twop_orbit[iconf][isrc][tsink][ireim] + twop_orbit[iconf][isrc][tsink2][ireim] );
            }
            data[iconf][T_global] /= (double)num_src_per_conf;

            data[iconf][T_global+1] = 0.;
            for ( int it = 0; it < T_global; it++ ) {
              data[iconf][T_global+1] += loop_sub[iconf][it];
            }
            data[iconf][T_global+1] /= (double)T_global;
          }
    
          char obs_name[100];
#ifdef _XG_NUCLEON
          sprintf ( obs_name, "ratio.sub.gf_%s.gi_%s.g4_D4.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s",
              gamma_id_to_Cg_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_Cg_ascii[ g_source_gamma_id_list[igi] ],
              smearing_tag,
              g_sequential_source_timeslice_list[idt],
              g_sink_momentum_list[0][0],
              g_sink_momentum_list[0][1],
              g_sink_momentum_list[0][2], reim_str[ireim] );
#else
          sprintf ( obs_name, "ratio.sub.gf_%s.gi_%s.g4_D4.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s",
              gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ],
              smearing_tag,
              g_sequential_source_timeslice_list[idt],
              g_sink_momentum_list[0][0],
              g_sink_momentum_list[0][1],
              g_sink_momentum_list[0][2], reim_str[ireim] );
#endif   
          /* UWerr parameters */
          int narg          = 3;
          int arg_first[3]  = { 0, T_global, T_global+1 };
          int arg_stride[3] = { 1,  0 , 0};

          /* apply UWerr analysis */
          exitstatus = apply_uwerr_func ( data[0], num_conf, T_global+2, T_global, narg, arg_first, arg_stride, obs_name, ratio_1_2_mi_3, dratio_1_2_mi_3 );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
    
          fini_2level_dtable ( &data );
        }  /* end of loop on reim */
    
#endif  /* end of ifdef _RAT_SUB_METHOD */

        fini_4level_dtable ( &threep_44 );
    
      }  /* end of loop on dt */


#endif  /* end of ifdef _RAT_METHOD */
  
      /**********************************************************/
      /**********************************************************/
  
      fini_2level_dtable ( &loop_sub );

#endif  /* end of ifdef _LOOP_ANALYSIS */

    }  /* end of loop on smearing levels */

    fini_6level_dtable ( &twop );
    fini_4level_dtable ( &twop_orbit );

  }}  /* end of loop on gi, gf */

  /**********************************************************
   * free and finalize
   **********************************************************/

  fini_3level_itable ( &conf_src_list );

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
    fprintf(stdout, "# [xg_analyse] %s# [xg_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [xg_analyse] %s# [xg_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);
}