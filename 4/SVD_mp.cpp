/************************************************************************************************/
/* SVD Using Jacobis Rotations									*/
/*												*/
/* Compile: g++ -O3 SVD.cpp -o SVD								*/
/* Arguments:											*/
/*												*/
/*	M = # of columns									*/
/*	N = # of Rows										*/
/*												*/
/*	Matrix must be squared (M=N)								*/
/*												*/
/*	-t = print out Timing and # of Iterations						*/
/*	-p = print out Results (U, S, V)							*/
/*	-d = Generate the Octave files for debug and verify correctness				*/
/*												*/
/* Use:	./SVD M N -t -p -d									*/
/*												*/
/* All arguments aren't important, just M and N. If you want, is possible to do 		*/
/* ./SVD M N -t and only print out the timing. As well you can use ./SVD M N -d for debug.	*/
/************************************************************************************************/

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sys/time.h>

#define epsilon 1.e-8

using namespace std;

template <typename T> double sgn(T val)
{
		return (val > T(0)) - (val < T(0));
}

int main (int argc, char* argv[]){

		int M,N;

		string T,P,Db;
		M = atoi(argv[1]);
		N = atoi(argv[2]);

		double elapsedTime,elapsedTime2;
		timeval start,end,end2;

		if(argc < 4){
				cout<<"Please input the size of Matrix and at least one of the options: -t -p -d";
				return 0;
		}


		if(M != N){
				cout<<"Error: Matrix must be square";
				return 0;
		}

		if(argc > 3){

				T = argv[3];
				if(argc > 4){
						P = argv[4];
						if(argc > 5){
								Db = argv[5];
						}
				}
		}
		// cout<<T<<P<<endl;

		double **U,**V, *S,**U_t, **V_t, **A;
		double alpha, beta, gamma, c, zeta, t,s,sub_zeta, converge;

		int acum = 0;
		int temp1, temp2;
		converge = 1.0;

		U = new double*[N];
		V = new double*[N];
		U_t = new double*[N];
		V_t = new double*[N];
		A = new double*[N];
		S = new double[N];

		for(int i =0; i<N; i++){
				U[i] = new double[N];
				V[i] = new double[N];
				U_t[i] = new double[N];
				V_t[i] = new double[N];
				A[i] = new double[N];
		}


		//Read from file matrix, if not available, app quit
		//Already transposed

		ifstream matrixfile("matrix");
		if(!(matrixfile.is_open())){
				cout<<"Error: file not found"<<endl;
				return 0;
		}

		for(int i = 0; i < M; i++){
				for(int j =0; j < N; j++){

						matrixfile >> U_t[i][j];
				}
		}

		matrixfile.close();


		for(int i=0; i<M;i++){
				for(int j=0; j<N;j++){

						if(i==j){
								V_t[i][j] = 1.0;
						}
						else{
								V_t[i][j] = 0.0;
						}
				}
		}


		//Store A for debug purpouse


		for(int i=0; i<M;i++){
				for(int j=0; j<N;j++){

						A[i][j] = U_t[j][i];
				}
		}





		/* SVD using Jacobi algorithm (Sequencial)*/

		gettimeofday(&start, NULL);

		double conv;
		int i, j, k;
		while(converge > epsilon){ 		//convergence
				converge = 0.0;	

				acum++;				//counter of loops

				for(i = 1; i<M; i++){
						for(j = 0; j<i; j++){


								alpha = 0.0;
								beta = 0.0;
								gamma = 0.0;

//								/*
								#pragma omp parallel for schedule(static, 125) private(k)  shared(N, i, j, U_t) \
							   		reduction(+:alpha, beta, gamma)
											for(k = 0; k<N ; k++){
													alpha = alpha + (U_t[i][k] * U_t[i][k]);
													beta = beta + (U_t[j][k] * U_t[j][k]);
													gamma = gamma + (U_t[i][k] * U_t[j][k]);
											}
//								*/
								/*
								#pragma omp parallel shared(i, j, U_t, alpha, beta, gamma) private(k)
								{
									#pragma omp sections nowait
									{

										#pragma omp section
										for(k = 0; k<N ; k++){
												alpha = alpha + (U_t[i][k] * U_t[i][k]);
										}
										#pragma omp section
										for(k = 0; k<N ; k++){
												beta = beta + (U_t[j][k] * U_t[j][k]);
										}
										#pragma omp section
										for(k = 0; k<N ; k++){
												gamma = gamma + (U_t[i][k] * U_t[j][k]);
										}
									}
								}
								*/
								converge = max(converge, abs(gamma)/sqrt(alpha*beta));	//compute convergence
								//basicaly is the angle
								//between column i and j


								zeta = (beta - alpha) / (2.0 * gamma);
								t = sgn(zeta) / (abs(zeta) + sqrt(1.0 + (zeta*zeta)));        //compute tan of angle
								c = 1.0 / (sqrt (1.0 + (t*t)));				//extract cos
								s = c*t;							//extrac sin


								//Apply rotations on U and V
								if (i != j) {
										#pragma omp parallel for schedule(static, 125)  private(k, t) shared(c, s, U_t, V_t) 
										for(k=0; k<N; k++){
												t = U_t[i][k];
												U_t[i][k] = c*t - s*U_t[j][k];
												U_t[j][k] = s*t + c*U_t[j][k];

												t = V_t[i][k];
												V_t[i][k] = c*t - s*V_t[j][k];
												V_t[j][k] = s*t + c*V_t[j][k];

										}

								} else {
										for(k=0; k<N; k++){
												t = U_t[i][k];
												U_t[i][k] = c*t - s*U_t[j][k];
												U_t[j][k] = s*t + c*U_t[j][k];

												t = V_t[i][k];
												V_t[i][k] = c*t - s*V_t[j][k];
												V_t[j][k] = s*t + c*V_t[j][k];

										}
								}

						}
				}
		}


		//Create matrix S

#pragma omp parallel for private(i, j, t) shared(U_t, S)
		for(int i =0; i<M; i++){

				t=0;
				for(int j=0; j<N;j++){
						t=t + pow(U_t[i][j],2);
				}
				t = sqrt(t);

				for(int j=0; j<N;j++){
						U_t[i][j] = U_t[i][j] / t;
						if(i == j){
								S[i] = t;
						}
				}
		}

		gettimeofday(&end, NULL);
		/************************************************************/

		/* Develop SVD Using OpenMP */



		// fix final result

		for(int i =0; i<M; i++){

				for(int j =0; j<N; j++){

						U[i][j] = U_t[j][i];
						V[i][j] = V_t[j][i];

				}

		}


		//Output time and iterations


		if(T=="-t" || P =="-t"){
				cout<<"iterations: "<<acum<<endl;
				elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
				elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
				cout<<"Time: "<<elapsedTime<<" ms."<<endl<<endl;


		}


		// Output the matrixes for debug
		if(T== "-p" || P == "-p"){
				cout<<"U"<<endl<<endl;
				for(int i =0; i<M; i++){
						for(int j =0; j<N; j++){

								cout<<U[i][j]<<"  ";
						}
						cout<<endl;
				}

				cout<<endl<<"V"<<endl<<endl;
				for(int i =0; i<M; i++){
						for(int j =0; j<N; j++){

								cout<<V[i][j]<<"  ";
						}
						cout<<endl;
				}

				cout<<endl<<"S"<<endl<<endl;
				for(int i =0; i<M; i++){
						for(int j =0; j<N; j++){

								if(i==j){  cout<<S[i]<<"  ";}

								else{
										cout<<"0.0  ";
								}
						}
						cout<<endl;
				}

		}

		//Generate Octave files for debug purpouse
		if(Db == "-d" || T == "-d" || P == "-d"){


				ofstream Af;
				//file for Matrix A
				Af.open("matrixAmp.mat"); 
				Af<<"# Created from debug\n# name: A\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";


				for(int i = 0; i<M;i++){
						for(int j =0; j<N;j++){
								Af<<" "<<A[i][j];
						}
						Af<<"\n";
				}

				Af.close();

				ofstream Uf;

				//File for Matrix U
				Uf.open("matrixUmp.mat");
				Uf<<"# Created from debug\n# name: Ucpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";

				for(int i = 0; i<M;i++){
						for(int j =0; j<N;j++){
								Uf<<" "<<U[i][j];
						}
						Uf<<"\n";
				}
				Uf.close();

				ofstream Vf;
				//File for Matrix V
				Vf.open("matrixVmp.mat");
				Vf<<"# Created from debug\n# name: Vcpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";

				for(int i = 0; i<M;i++){
						for(int j =0; j<N;j++){
								Vf<<" "<<V[i][j];
						}
						Vf<<"\n";
				}


				Vf.close();

				ofstream Sf;
				//File for Matrix S
				Sf.open("matrixSmp.mat");
				Sf<<"# Created from debug\n# name: Scpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";


				for(int i = 0; i<M;i++){
						for(int j =0; j<N;j++){
								if(i == j){
										Sf<<" "<<S[i];

								}

								else{
										Sf<<" 0.0";
								}
						}
						Sf<<"\n";
				}


				Sf.close();


		}

		delete [] S;
		for(int i = 0; i<N;i++){
				delete [] A[i];
				delete [] U[i];
				delete [] V[i];
				delete [] U_t[i];
				delete [] V_t[i];
		}

		return 0;
}






