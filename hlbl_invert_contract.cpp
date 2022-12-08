/****************************************************
 * hlbl_invert_contract
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
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

#  ifdef HAVE_KQED
#    include "KQED.h"
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

#include "clover.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform contractions for hlbl tensor\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {

  /*                            gt  gx  gy  gz */
  int const gamma_v_list[4] = {  0,  1,  2,  3 };
  int const gamma_v_num = 4;

  int const ymax = 2;
  int const ysign_num = 2;
  int const ysign_comb[16][4] = {
    { 1, 1, 1, 1},
    { 1, 1, 1,-1},
    { 1, 1,-1, 1},
    { 1, 1,-1,-1},
    { 1,-1, 1, 1},
    { 1,-1, 1,-1},
    { 1,-1,-1, 1},
    { 1,-1,-1,-1},
    {-1, 1, 1, 1},
    {-1, 1, 1,-1},
    {-1, 1,-1, 1},
    {-1, 1,-1,-1},
    {-1,-1, 1, 1},
    {-1,-1, 1,-1},
    {-1,-1,-1, 1},
    {-1,-1,-1,-1}
  };

  int idx_comb[6][2] = {
        {0,1},
        {0,2},
        {0,3},
        {1,2},
        {1,3},
        {2,3} };

  int c;
  int filename_set = 0;
  int gsx[4];
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  unsigned int Vhalf;
  char filename[400];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  int check_position_space_WI = 0;
  int first_solve_dummy = 1;
  struct timeval start_time, end_time;


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

  gettimeofday ( &start_time, (struct timezone *)NULL );

  /* set the default values */
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [hlbl_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [hlbl_invert_contract] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
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

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [hlbl_invert_contract] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hlbl_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hlbl_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hlbl_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[hlbl_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  Vhalf                  = VOLUME / 2;
  size_t sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);
  size_t sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [hlbl_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [hlbl_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[hlbl_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[hlbl_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[hlbl_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[hlbl_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[hlbl_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************
   * set io process
   ***********************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[hlbl_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [hlbl_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***********************************************************
   ***********************************************************
   **
   ** dummy inversion for solver tuning
   **
   ** use volume source
   **
   ***********************************************************
   ***********************************************************/

  if ( first_solve_dummy )
  {
    /***********************************************************
     * initialize rng state
     ***********************************************************/
    exitstatus = init_rng_stat_file ( g_seed, NULL );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[hlbl_invert_contract] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
      EXIT( 50 );
    }
  
    double ** spinor_field = init_2level_dtable ( 2, _GSI( (size_t)VOLUME ));
    if( spinor_field == NULL ) {
      fprintf(stderr, "[hlbl_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    double ** spinor_work = init_2level_dtable ( 2, _GSI( (size_t)(VOLUME+RAND) ));
    if( spinor_work == NULL ) {
      fprintf(stderr, "[hlbl_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    if( ( exitstatus = prepare_volume_source ( spinor_field[0], VOLUME ) ) != 0 ) {
      fprintf(stderr, "[hlbl_invert_contract] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(64);
    }

    memcpy ( spinor_work[0], spinor_field[0], sizeof_spinor_field );
    memset ( spinor_work[1], 0, sizeof_spinor_field );

    /* full_spinor_work[1] = D^-1 full_spinor_work[0],
     * flavor id 0 
     */
    exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], 0 );
    if(exitstatus < 0) {
      fprintf(stderr, "[hlbl_invert_contract] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }

    /* check residuum */
    exitstatus = check_residual_clover (&(spinor_work[1]) , &(spinor_work[0]), gauge_field_with_phase, mzz[0], mzzinv[0], 1);

    fini_2level_dtable ( &spinor_work );
    fini_2level_dtable ( &spinor_field );

  }  /* end of first_solve_dummy */

#if 0
  /***********************************************************
   * test teh QED Kernel package linking 
   ***********************************************************/
  struct QED_kernel_temps kqed_t ;

  if( initialise( &kqed_t ) ) 
  {
    fprintf(stderr, "[hlbl_invert_contract] Error from kqed initialise, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(19);
  }

  FILE * fs = NULL;
  sprintf ( filename, "kqed.proc%.4d.test", g_cart_id );
  fs = fopen( filename, "w" );

  double const AlfQED = 0.00729735256981631454;

  static const double Mv = 2.0 ; // ratio of lepton-loop mass to muon mass
  const double pref = (pow(4.0*M_PI*AlfQED,3)/3)*(2.0*M_PI*M_PI)*(4*M_PI);
  const double convf = pow(Mv,7);
  const double norm_fermloop = pow(1.0,4);
  const double fnorm = 2.0*pref*convf*norm_fermloop ;

  double kerv[6][4][4][4] KQED_ALIGN ;
  double pihat[6][4][4][4] KQED_ALIGN ;

  double x,y,z ;
  for( x = 0.005 ; x < 1 ; x += 0.2 ) {
    for( y = 0.005 ; y < M_PI ; y += M_PI/4. ) {
      for( z = 0.005 ; z < 1 ; z += 0.2 ) {

        const double vi[3] = { x , y , z } ;

        const double co = cos(vi[1]);
        const double si = sin(vi[1]);

        const double xv[4] = { 0 , 0 , vi[0]*si , vi[0]*co } ;
        const double yv[4] = { 0 , 0 , 0 , vi[2] } ;

        const double yvMv[4] = { 0 , 0 , 0 , vi[2]*Mv } ;
        const double xvMv[4] = { 0 , 0 , vi[0]*si*Mv , vi[0]*co*Mv } ;

        QED_kernel_L0( xv, yv, kqed_t, kerv ) ;

        ipihatFermLoop_antisym( xvMv, yvMv, kqed_t, pihat );

        const double *pi = (const double*)pihat ;
        const double *kp = (const double*)kerv ;
        register double tmp = 0.0;
        int idx ;
        for( idx = 0 ; idx < 384 ; idx++ ) {
          tmp += *pi * ( *kp ) ;
          pi++ ; kp++ ;
        }
        tmp *= fnorm;
        fprintf( fs, "x= %lf beta= %lf  y= %lf  iPihat*L_QED= %.11lg\n",
                 vi[0], vi[1], vi[2], 2*pow(vi[0],3.0)*pow(vi[2],4.0)*si*si*tmp);
      }
    }
  }

  fclose ( fs) ;
#endif

  double ** spinor_work = init_2level_dtable ( 2, _GSI( (size_t)(VOLUME+RAND) ));
  if( spinor_work == NULL ) {
    fprintf(stderr, "[hlbl_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }

  double *** fwd_src = init_3level_dtable ( 2, 12, _GSI( (size_t)VOLUME ) );
  
  double *** fwd_y   = init_3level_dtable ( 2, 12, _GSI( (size_t)VOLUME ) );

  double **** seq_src = init_4level_dtable ( 2, 6, 12, _GSI( (size_t)VOLUME ) );
  
  double *** seq_y   = init_3level_dtable ( 6, 12, _GSI( (size_t)VOLUME ) );

  if( fwd_src == NULL || fwd_y == NULL || seq_src == NULL || seq_y == NULL )
  {
    fprintf(stderr, "[hlbl_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }



  for ( int isrc = 0; isrc < g_source_location_number; isrc++ )
  {

    /***********************************************************
     * determine source coordinates, find out, if source_location is in this process
     ***********************************************************/
    int gsx[4], sx[4];
    gsx[0] = ( g_source_coords_list[isrc][0] +  T_global ) %  T_global;
    gsx[1] = ( g_source_coords_list[isrc][1] + LX_global ) % LX_global;
    gsx[2] = ( g_source_coords_list[isrc][2] + LY_global ) % LY_global;
    gsx[3] = ( g_source_coords_list[isrc][3] + LZ_global ) % LZ_global;

    int source_proc_id = -1;
    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

   /***********************************************************
    * forward proapgators from source
    ***********************************************************/

    for ( int iflavor = 0; iflavor <= 1; iflavor ++ ) 
    {
      for ( int i = 0; i < 12; i++ ) 
      {
        memset ( spinor_work[0], 0, sizeof_spinor_field );
        memset ( spinor_work[1], 0, sizeof_spinor_field );

        if ( source_proc_id == g_cart_id ) 
        {
          spinor_work[0][_GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]]) + 2*i ] = 1.;
        }

        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );

        if(exitstatus < 0) {
          fprintf(stderr, "[hlbl_invert_contract] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(19);
        }
 
        /* check residuum */
        exitstatus = check_residual_clover (&(spinor_work[1]) , &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1);
        if(exitstatus != 0) {
          fprintf(stderr, "[hlbl_invert_contract] Error from check_residual_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(19);
        }

        memcpy ( fwd_src[iflavor][i], spinor_work[1], sizeof_spinor_field );
     
        if ( g_write_propagator ) 
        {
          sprintf ( filename, "fwd_0.f%d.t%dx%dy%dz%d.sc%d.lime", iflavor, gsx[0] , gsx[1] ,gsx[2] , gsx[3], i );

          if ( ( exitstatus = write_propagator( spinor_work[1], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[hlbl_invert_contract] Error from write_propagator for %s, status was %d   %s %d\n", filename, exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }
      
      }  /* end of loop on spin-color components */

    }  /* end of loop on flavor */

    /***********************************************************
     * seq_src 
     ***********************************************************/
    for ( int iflavor = 0; iflavor <= 1; iflavor ++ ) 
    {
      for ( int k = 0; k < 6; k++ ) 
      {

        for ( int i = 0; i < 12; i++ ) 
        {
          memset ( spinor_work[0], 0, sizeof_spinor_field );
          memset ( spinor_work[1], 0, sizeof_spinor_field );

#pragma omp parallel for
          for ( unsigned int ix = 0; ix < VOLUME; ix++ )
          {
            double * const _s = spinor_work[0] + _GSI(ix);
            double * const _r = fwd_src[iflavor][i] + _GSI(ix);
            double sp[24], sp2[24];

            int const z[4] = {
              g_lexic2coords[ix][0] + g_proc_coords[0] * T,
              g_lexic2coords[ix][1] + g_proc_coords[1] * LX,
              g_lexic2coords[ix][2] + g_proc_coords[2] * LY,
              g_lexic2coords[ix][3] + g_proc_coords[3] * LZ };

            _fv_eq_gamma_ti_fv ( sp,  idx_comb[k][0], _r );
            _fv_eq_gamma_ti_fv ( sp2, idx_comb[k][1], _r );

            _fv_ti_eq_re ( sp,   z[idx_comb[k][1]] );
            _fv_ti_eq_re ( sp2, -z[idx_comb[k][0]] );

            _fv_eq_fv_pl_fv ( _s, sp, sp2 );
          }

          exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );

          if(exitstatus < 0) {
            fprintf(stderr, "[hlbl_invert_contract] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(19);
          }
 
          /* check residuum */
          exitstatus = check_residual_clover (&(spinor_work[1]) , &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1);
          if(exitstatus != 0) {
            fprintf(stderr, "[hlbl_invert_contract] Error from check_residual_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(19);
          }

          memcpy ( seq_src[iflavor][k][i], spinor_work[1], sizeof_spinor_field );
     
          if ( g_write_propagator ) 
          {
            sprintf ( filename, "seq_0.f%d.t%dx%dy%dz%d.z%d.g%d.sc%d.lime", iflavor, gsx[0] , gsx[1] ,gsx[2] , gsx[3], idx_comb[k][0], idx_comb[k][1], i );

            if ( ( exitstatus = write_propagator( spinor_work[1], filename, 0, g_propagator_precision) ) != 0 ) {
              fprintf(stderr, "[hlbl_invert_contract] Error from write_propagator for %s, status was %d   %s %d\n", filename, exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }
          }
      
        }  /* end of loop on spin-color components */

      }  /* end of loop on antisymmetric index combinations */

    }  /* end of loop on flavor */

    for ( int iflavor = 0; iflavor <= 1; iflavor ++ ) 
    {
 
      /***********************************************************
       * loop on y = iy ( 1,1,1,1)
       ***********************************************************/
      for ( int iy = 1; iy <= ymax; iy++ )
      {

        /***********************************************************
         * loop on directions in 4-space
         ***********************************************************/
        for ( int isign = 0; isign < ysign_num; isign++ )
        {

          int gsx[4], sx[4];
          gsx[0] = ( iy * ysign_comb[isign][0] + g_source_coords_list[isrc][0] +  T_global ) %  T_global;
          gsx[1] = ( iy * ysign_comb[isign][1] + g_source_coords_list[isrc][1] + LX_global ) % LX_global;
          gsx[2] = ( iy * ysign_comb[isign][2] + g_source_coords_list[isrc][2] + LY_global ) % LY_global;
          gsx[3] = ( iy * ysign_comb[isign][3] + g_source_coords_list[isrc][3] + LZ_global ) % LZ_global;

          int source_proc_id = -1;
          exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
          if( exitstatus != 0 ) {
            fprintf(stderr, "[p2gg_invert_contract_local] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(123);
          }

          /***********************************************************
           * forward proapgators from y
           ***********************************************************/
  
          for ( int i = 0; i < 12; i++ ) 
          {
            memset ( spinor_work[0], 0, sizeof_spinor_field );
            memset ( spinor_work[1], 0, sizeof_spinor_field );
      
            if ( source_proc_id == g_cart_id ) 
            {
              spinor_work[0][_GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]]) + 2*i ] = 1.;
            }
      
            exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
    
            if(exitstatus < 0) {
              fprintf(stderr, "[hlbl_invert_contract] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(19);
            }
       
            /* check residuum */
            exitstatus = check_residual_clover (&(spinor_work[1]) , &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1);
            if(exitstatus != 0) {
              fprintf(stderr, "[hlbl_invert_contract] Error from check_residual_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(19);
            }
      
            memcpy ( fwd_y[iflavor][i], spinor_work[1], sizeof_spinor_field );
           
            if ( g_write_propagator ) 
            {
              sprintf ( filename, "fwd_y.f%d.t%dx%dy%dz%d.y%d.st%dsx%dsy%dsz%d.sc%d.lime", iflavor, gsx[0] , gsx[1] ,gsx[2] , gsx[3], iy,
                ysign_comb[isign][0], ysign_comb[isign][1], ysign_comb[isign][2], ysign_comb[isign][3], i );
    
              if ( ( exitstatus = write_propagator( spinor_work[1], filename, 0, g_propagator_precision) ) != 0 ) {
                fprintf(stderr, "[hlbl_invert_contract] Error from write_propagator for %s, status was %d   %s %d\n", filename, exitstatus, __FILE__, __LINE__);
                EXIT(2);
              }
            }
            
          }  /* end of loop on spin-color components */
      
          /***********************************************************
           * seq_y
           ***********************************************************/
          for ( int k = 0; k < 6; k++ ) 
          {
    
            for ( int i = 0; i < 12; i++ ) 
            {
              memset ( spinor_work[0], 0, sizeof_spinor_field );
              memset ( spinor_work[1], 0, sizeof_spinor_field );
    
    #pragma omp parallel for
              for ( unsigned int ix = 0; ix < VOLUME; ix++ )
              {
                double * const _s = spinor_work[0] + _GSI(ix);
                double * const _r = fwd_y[iflavor][i] + _GSI(ix);
                double sp[24], sp2[24];
    
                int const z[4] = {
                  g_lexic2coords[ix][0] + g_proc_coords[0] * T,
                  g_lexic2coords[ix][1] + g_proc_coords[1] * LX,
                  g_lexic2coords[ix][2] + g_proc_coords[2] * LY,
                  g_lexic2coords[ix][3] + g_proc_coords[3] * LZ };
    
                _fv_eq_gamma_ti_fv ( sp,  idx_comb[k][0], _r );
                _fv_eq_gamma_ti_fv ( sp2, idx_comb[k][1], _r );
    
                _fv_ti_eq_re ( sp,   z[idx_comb[k][1]] );
                _fv_ti_eq_re ( sp2, -z[idx_comb[k][0]] );
    
                _fv_eq_fv_pl_fv ( _s, sp, sp2 );
              }
    
              exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
    
              if(exitstatus < 0) {
                fprintf(stderr, "[hlbl_invert_contract] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(19);
              }
     
              /* check residuum */
              exitstatus = check_residual_clover (&(spinor_work[1]) , &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1);
              if(exitstatus != 0) {
                fprintf(stderr, "[hlbl_invert_contract] Error from check_residual_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(19);
              }
    
              memcpy ( seq_y[k][i], spinor_work[1], sizeof_spinor_field );
         
              if ( g_write_propagator ) 
              {
                sprintf ( filename, "seq_y.f%d.t%dx%dy%dz%d.z%d.g%d.y%d.st%dsx%dsy%dsz%d.sc%d.lime", iflavor, gsx[0] , gsx[1] ,gsx[2] , gsx[3], idx_comb[k][0], idx_comb[k][1],
                   iy, ysign_comb[isign][0], ysign_comb[isign][1], ysign_comb[isign][2], ysign_comb[isign][3], i );
    
                if ( ( exitstatus = write_propagator( spinor_work[1], filename, 0, g_propagator_precision) ) != 0 ) {
                  fprintf(stderr, "[hlbl_invert_contract] Error from write_propagator for %s, status was %d   %s %d\n", filename, exitstatus, __FILE__, __LINE__);
                  EXIT(2);
                }
              }
          
            }  /* end of loop on spin-color components */
    
          }  /* end of loop on antisymmetric index combinations */


        }  /* end of loop on signs */

      }  /* end of loop on |y| */

    }  /* end of loop on flavor */

  }  /* end of loop on source locations */


  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/
 
  fini_3level_dtable ( &fwd_src );
  fini_3level_dtable ( &fwd_y );
  fini_4level_dtable ( &seq_src );
  fini_3level_dtable ( &seq_y );
  fini_2level_dtable ( &spinor_work );


#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  /* free clover matrix terms */
  fini_clover ( &mzz, &mzzinv );

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
  show_time ( &start_time, &end_time, "hlbl_invert_contract", "runtime", g_cart_id == 0 );

  return(0);
}
