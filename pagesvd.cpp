/*
 * Page-wise matrix SVD decomposition based on LAPACK and OMP threads.
 * 
 * Syntax:
 *          S = pagesvd(X)
 *          [U S V] = pagesvd(X)
 *          [__] = pagesvd(X,'econ')
 *          [__] = pagesvd(__,outputForm)
 *
 *          outputForm can be 'matrix' or 'vector' [or 'trans' = same as 'matrix' but returns V' not V]
 *
 * To compile:
 *          mex pagesvd.cpp -v -lmwlapack -R2018a
 */

#include "mex.h"
#include "lapack.h"
#include <complex>
#include <string.h>
#include <omp.h>
#include <algorithm>
#include <cassert>
#include <iostream>

#if !MX_HAS_INTERLEAVED_COMPLEX
#error "This MEX-file must be compiled with the -R2018a flag."
#endif

char* lower(char *buf);

template <typename T>
inline void transpose(T *A, long m, long n);

template <typename T>
inline void conjugate(T *A, long m, long n);

template <typename T>
inline void shift_rows(T *S, long m, long n);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* check arguments */
    if (nrhs > 3) mexErrMsgTxt("Too many input arguments.");    
    if (nlhs > 3) mexErrMsgTxt("Too many output arguments.");          
    if (nrhs < 1) mexErrMsgTxt("Not enough input arguments.");
    if (mxGetClassID(prhs[0])!=mxSINGLE_CLASS && mxGetClassID(prhs[0])!=mxDOUBLE_CLASS) mexErrMsgTxt("First argument must be numeric array.");
    
    char jobz       = (nlhs > 1) ? 'A' : 'N'; // A(ll) or S(mall) [N(o U or V): singular values only
    char outputForm = (nlhs > 1) ? 'M' : 'V'; // M(atrix) or V(ector) [T(rans): same as M but returns V' not V]
    bool trans = false; // outputForm = 'T' returns the transpose V' instead of V (faster as lapack returns V')
    
    if (nrhs>1)
    {
        char buf[8];
        if (mxGetString(prhs[1],buf,5)==0 && strncmp(lower(buf),"econ",5)==0)
        {
            if (nlhs > 1) jobz = 'S';
        }
        else if (mxGetString(prhs[1],buf,7)==0 && strncmp(lower(buf),"vector",7)==0)
            outputForm = 'V';
        else if (mxGetString(prhs[1],buf,7)==0 && strncmp(lower(buf),"matrix",7)==0)
            outputForm = 'M';      
        else if (mxGetString(prhs[1],buf,6)==0 && strncmp(lower(buf),"trans",6)==0)
        {
            trans = true;
            outputForm = 'M';
        }
        else
            mexErrMsgTxt("Second input argument must be 'econ', vector' or 'matrix'.");
        
        if (nrhs>2)
        {
            if (mxGetString(prhs[2],buf,7)==0 && strncmp(lower(buf),"vector",7)==0)
                outputForm = 'V';
            else if (mxGetString(prhs[2],buf,7)==0 && strncmp(lower(buf),"matrix",7)==0)
                outputForm = 'M';       
            else if (mxGetString(prhs[2],buf,6)==0 && strncmp(lower(buf),"trans",6)==0)
            {
                trans = true;
                outputForm = 'M';
            }
            else
                mexErrMsgTxt("Third input argument must be 'matrix' or 'vector'.");

            if (jobz=='A') mexErrMsgTxt("Cannot specify 'matrix' and 'vector'.");
        }
    }

    /* duplicate input array (overwritten by lapack routines) */  
    mxArray *a = mxDuplicateArray(prhs[0]);
    if (!a) mexErrMsgTxt("Insufficent memory (a).");

    /* check matrix */  
    mwSize ndim = mxGetNumberOfDimensions(a);
    const mwSize *adim = mxGetDimensions(a);
    
    ptrdiff_t m = ndim>0 ? adim[0] : 1;
    ptrdiff_t n = ndim>1 ? adim[1] : 1;
    ptrdiff_t p = ndim>2 ? adim[2] : 1;
    for (long i = 3; i < ndim; i++) p *= adim[i]; /* stack all higher dimensions */
    
    ptrdiff_t mn = std::min(m,n);
    ptrdiff_t mx = std::max(m,n);

    /* output arrays: s, u, v */   
    ptrdiff_t sdim[3] = {m,n,p}; 
    if (jobz=='S' || jobz=='N') sdim[0] = sdim[1] = mn;
    if (outputForm=='V') sdim[1] = 1;

    ptrdiff_t udim[3] = {m,m,p};
    if (jobz=='S') udim[1] = mn;
    if (jobz=='N') udim[0] = udim[1] = 1;

    ptrdiff_t vdim[3] = {n,n,p};
    if (jobz=='S') vdim[0] = mn;
    if (jobz=='N') vdim[0] = vdim[1] = 1;
    
    mwSize xdim[std::max<mwSize>(ndim,3)]; /* for copying ptrdiff_t[] into mwSize[] */
    mxArray *s = mxCreateNumericArray(3, std::copy_n(sdim,3,xdim)-3, mxGetClassID(a), mxREAL);    
    mxArray *u = mxCreateNumericArray(3, std::copy_n(udim,3,xdim)-3, mxGetClassID(a), mxIsComplex(a) ? mxCOMPLEX : mxREAL);    
    mxArray *v = mxCreateNumericArray(3, std::copy_n(vdim,3,xdim)-3, mxGetClassID(a), mxIsComplex(a) ? mxCOMPLEX : mxREAL);
    
    if (!s || !u || !v) mexErrMsgTxt("Insufficent memory (s, u, v).");
   
    /* Get the number of threads from the Matlab engine (maxNumCompThreads) */
    mxArray *matlabCallOut[1] = {0};
    mxArray *matlabCallIn[1]  = {0};
    mexCallMATLAB(1, matlabCallOut, 0, matlabCallIn, "maxNumCompThreads");
    double *pthreads = mxGetPr(matlabCallOut[0]);
    int nthreads = int(*pthreads);
    if (nthreads == 1) mexWarnMsgTxt("pagesvd threads equals 1. Try increasing maxNumCompThreads().");
    
/* run in parallel on single cores */
#pragma omp parallel num_threads(nthreads)
if (m*n*p)
{ 
    /* workspace calculations */
    ptrdiff_t *iwork = (ptrdiff_t*)mxMalloc( 8 * mn * sizeof(ptrdiff_t) );
      
    ptrdiff_t sz = std::max(5*mn*mn+5*mn,2*mx*mn+2*mn*mn+mn);
    void *rwork = (void*)mxMalloc( sz * mxGetElementSize(a) );
    
    ptrdiff_t lwork = std::max(mn*mn+2*mn+mx,4*mn*mn+6*mn+mx);
    void *work = (void*)mxMalloc( lwork * mxGetElementSize(a) );
    
    if (!iwork || !rwork || !work) mexErrMsgTxt("Insufficent memory (work).");   
    
    /* svd and ctranspose v */   
    #pragma omp for
    for (long i = 0; i < p; i++)
    {  
        ptrdiff_t info;
        
        /* pointers to the i-th matrix (use char* to suppress compiler warnings */
        void *a_i = (char*)mxGetData(a) + i * adim[0] * adim[1] * mxGetElementSize(a);
        void *s_i = (char*)mxGetData(s) + i * sdim[0] * sdim[1] * mxGetElementSize(s);
        void *u_i = (char*)mxGetData(u) + i * udim[0] * udim[1] * mxGetElementSize(u);
        void *v_i = (char*)mxGetData(v) + i * vdim[0] * vdim[1] * mxGetElementSize(v);

        // real float
        if(!mxIsComplex(a) && !mxIsDouble(a))
        {
            sgesdd(&jobz, &m, &n, (float*)a_i, &m, (float*)s_i, (float*)u_i, udim, (float*)v_i, vdim, (float*)work, &lwork, iwork, &info);
            
            if (trans==false) transpose((float*)v_i, vdim[0], vdim[1]);
            if (outputForm=='M') shift_rows((float*)s_i, sdim[0], sdim[1]);
        }
        
        // real double
        else if(!mxIsComplex(a) &&  mxIsDouble(a))
        {
            dgesdd(&jobz, &m, &n, (double*)a_i, &m, (double*)s_i, (double*)u_i, udim, (double*)v_i, vdim, (double*)work, &lwork, iwork, &info);
        
            if (trans==false) transpose((double*)v_i, vdim[0], vdim[1]);
            if (outputForm=='M') shift_rows((double*)s_i, sdim[0], sdim[1]);

        }   
        
        // complex float
        else if( mxIsComplex(a) && !mxIsDouble(a))
        {
            cgesdd(&jobz, &m, &n, (float*)a_i, &m, (float*)s_i, (float*)u_i, udim, (float*)v_i, vdim, (float*)work, &lwork, (float*)rwork, iwork, &info);

            if (trans==false) transpose((std::complex<float>*)v_i, vdim[0], vdim[1]);
            if (trans==false) conjugate((std::complex<float>*)v_i, vdim[0], vdim[1]);   
            if (outputForm=='M') shift_rows((float*)s_i, sdim[0], sdim[1]);      
        }
        
        // complex double
        else if( mxIsComplex(a) &&  mxIsDouble(a))
        {
            zgesdd(&jobz, &m, &n, (double*)a_i, &m, (double*)s_i, (double*)u_i, udim, (double*)v_i, vdim, (double*)work, &lwork, (double*)rwork, iwork, &info);
            
            if (trans==false) transpose((std::complex<double>*)v_i, vdim[0], vdim[1]);
            if (trans==false) conjugate((std::complex<double>*)v_i, vdim[0], vdim[1]);
            if (outputForm=='M') shift_rows((double*)s_i, sdim[0], sdim[1]);
        }
        
        if(info) mexErrMsgTxt("dgesdd failed (run).");
    
    } /* end of pragma omp for loop */
    
    mxFree(work);
    mxFree(iwork);
    mxFree(rwork);
    
} /* end of pragma omp parallel block */

    /* reshape to match input */
    std::copy_n(adim, ndim, xdim);
    if (trans==false) std::swap(vdim[0], vdim[1]);
    xdim[0] = sdim[0]; xdim[1] = sdim[1]; mxSetDimensions(s, xdim, ndim);    
    xdim[0] = udim[0]; xdim[1] = udim[1]; mxSetDimensions(u, xdim, ndim);
    xdim[0] = vdim[0]; xdim[1] = vdim[1]; mxSetDimensions(v, xdim, ndim);

    if (nlhs > 2)
    {
        plhs[0] = u;
        plhs[1] = s;
        plhs[2] = v;
    }
    else if (nlhs > 1)
    {
        plhs[0] = u;
        plhs[1] = s;
        mxDestroyArray(v);
    }
    else if (nlhs >= 0)
    {
        plhs[0] = s;
        mxDestroyArray(u);
        mxDestroyArray(v);
    }
    
    mxDestroyArray(a);
}
      

// In-place convert to lowercase
char* lower(char *buf)
{
    for (long i = 0; buf[i]; i++)
        buf[i] = std::tolower(buf[i]);
    
    return buf;
}


// In-place complex conjugate
template<typename T>
inline void conjugate(T *A, long m, long n)
{
    for (long i = 0; i < m*m; i++)
        A[i].imag(-A[i].imag());
}


// In-place matrix shift
template <typename T>
inline void shift_rows(T *S, long m, long n)
{
    for (long i = 1; i < std::min(m,n); i++)
        std::swap(S[i], S[i+m*i]);
}


// In-place matrix transpose
template<typename T>
inline void transpose(T *A, long m, long n)
{
    if (m < n) std::swap(m, n);

    while (m > 1 && n > 1)
    {
        for (long i = 1; i < m; i++)
            std::rotate(A+i, A+i*n, A+i*n+1);
        
        A += m;
        n -= 1;
    }
}
