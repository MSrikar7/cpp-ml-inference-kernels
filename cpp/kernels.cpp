#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Row-major: Y[M,N] = X[M,K] * W[K,N] + bias[N].
using Mat = std::vector<float>;
void reference(const Mat& x, const Mat& w, const Mat& bias, Mat& y, int m, int k, int n) {
    for (int i=0;i<m;++i) for(int j=0;j<n;++j) {
        float acc=bias[j];
        for(int p=0;p<k;++p) acc+=x[i*k+p]*w[p*n+j];
        y[i*n+j]=acc;
    }
}
void tiled(const Mat& x, const Mat& w, const Mat& bias, Mat& y, int m, int k, int n, int tile, bool parallel) {
    std::fill(y.begin(), y.end(), 0.0f);
    // Every (row-tile, column-tile) writes a disjoint part of Y.
    #pragma omp parallel for collapse(2) if(parallel) schedule(static)
    for(int ii=0; ii<m; ii+=tile) for(int jj=0; jj<n; jj+=tile) {
        int imax=std::min(m,ii+tile), jmax=std::min(n,jj+tile);
        for(int kk=0;kk<k;kk+=tile) {
            int kmax=std::min(k,kk+tile);
            for(int i=ii;i<imax;++i) for(int p=kk;p<kmax;++p) {
                float a=x[i*k+p];
                for(int j=jj;j<jmax;++j) y[i*n+j]+=a*w[p*n+j];
            }
        }
        for(int i=ii;i<imax;++i) for(int j=jj;j<jmax;++j) y[i*n+j]+=bias[j];
    }
}
int main(int argc,char** argv) {
    int m=128,k=256,n=128,tile=32,reps=5,threads=4;
    if(argc>1)m=std::stoi(argv[1]);if(argc>2)k=std::stoi(argv[2]);
    if(argc>3)n=std::stoi(argv[3]);if(argc>4)tile=std::stoi(argv[4]);
    if(argc>5)reps=std::stoi(argv[5]);if(argc>6)threads=std::stoi(argv[6]);
    if(m<=0||k<=0||n<=0||tile<=0||reps<=0||threads<=0) return 2;
    #ifdef _OPENMP
    omp_set_dynamic(0); omp_set_num_threads(threads);
    #endif
    std::mt19937 rng(42);std::uniform_real_distribution<float> dist(-0.25f,0.25f);
    Mat x(size_t(m)*k), w(size_t(k)*n), bias(n), y(size_t(m)*n), truth(size_t(m)*n);
    for(auto& a:x)a=dist(rng);for(auto& a:w)a=dist(rng);for(auto& a:bias)a=dist(rng);
    reference(x,w,bias,truth,m,k,n);
    std::cout << "method,m,k,n,tile,threads,median_ms,max_abs_error,gflops\n";
    for(const auto& method:{std::string("naive"),std::string("tiled"),std::string("tiled_parallel")}) {
        std::vector<double> times;
        for(int r=0;r<reps+1;++r) {
            auto start=std::chrono::steady_clock::now();
            if(method=="naive")reference(x,w,bias,y,m,k,n);
            else tiled(x,w,bias,y,m,k,n,tile,method=="tiled_parallel");
            auto end=std::chrono::steady_clock::now();
            if(r>0)times.push_back(std::chrono::duration<double,std::milli>(end-start).count());
        }
        std::sort(times.begin(),times.end());
        double median=times[times.size()/2];
        double err=0;
        for(size_t i=0;i<y.size();++i)err=std::max(err,double(std::abs(y[i]-truth[i])));
        double gflops=2.0*m*k*n/(median*1e6);
        std::cout<<method<<','<<m<<','<<k<<','<<n<<','<<tile<<','<<threads<<','
                 <<std::fixed<<std::setprecision(6)<<median<<','<<err<<','<<gflops<<'\n';
        if(err>0.002) return 1;
    }
}
