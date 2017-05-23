#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sys/time.h>
#include <mpi.h>

#define epsilon 1.e-8
#define X_LEN (nproc > 8 ? 4 : 2)	
#define P(x, y) (x * N + y)
#define PP(x, y) ((x - begin_x) * dis_y + (y - begin_y))
#define PPP(x, y) ((x / dis_x * len_y + y / dis_y) * dis + (x % dis_x) * dis_y + y % dis_y)

using namespace std;

#define PRINT_LINE printf("line: %d\n", __LINE__)

int main (int argc, char* argv[]){

		int M,N, nproc, myrank;
		int len_x, len_y;
		int begin_x, begin_y, end_x, end_y;
		int pos_x, pos_y;
		int dis_x, dis_y;
		string T,P,Db;
		M = atoi(argv[1]);
		N = atoi(argv[2]);
		nproc = atoi(argv[3]);

		MPI_Init(NULL, NULL);
		MPI_Comm_size(MPI_COMM_WORLD, &nproc);
		/* splite the data */
		len_x = X_LEN;
		len_y = nproc / X_LEN;
		dis_x = M / len_x;
		dis_y = M / len_y;	

		printf("X_LEN %d, Y_LEN %d\n", len_x, len_y);
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
//		printf("My rank %d\n", myrank);
		cout << "my_rank: " << myrank << endl;
		/* cal the position of this block */
		pos_x = myrank / len_y;
		pos_y = myrank % len_y;	
		begin_x = pos_x * dis_x;
		begin_y = pos_y * dis_y;
		end_x = (pos_x + 1) * dis_x;
		end_y = (pos_y + 1) * dis_y;

		cout << "begin_x: " << begin_x << ", begin_y: " << begin_y << endl;
//		printf("begin_x: %d, end_x: %d\n", begin_x, end_x);
//		printf("begin_y: %d, end_y: %d\n", begin_y, end_y);

		double elapsedTime,elapsedTime2;
		timeval start,end,end2;

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

		double *U_t;
		double alpha, beta, gamma,*Alphas,*Betas,*Gammas;

		int acum = 0;
		int temp1, temp2;

/*
		U_t = new double*[N];
		Alphas = new double*[N];
		Betas = new double*[N];
		Gammas = new double*[N];

		for(int i =0; i<N; i++){
				U_t[i] = new double[N];
				Alphas[i] = new double[N];
				Betas[i] = new double[N];
				Gammas[i] = new double[N];
		}
*/
		U_t = (double *)malloc(M * N * sizeof (double) );
		Alphas = (double *)malloc(M * M * sizeof (double) );
		Betas = (double *)malloc(M * M * sizeof (double) );
		Gammas = (double *)malloc(M * M * sizeof (double) );

		//Read from file matrix, if not available, app quit
		//Already transposed

		ifstream matrixfile("matrix");
		if(!(matrixfile.is_open())){
				cout<<"Error: file not found"<<endl;
				return 0;
		}

		for(int i = 0; i < M; i++){
				for(int j =0; j < N; j++){

						matrixfile >> U_t[P(i, j)];
				}
		}

		matrixfile.close();

		/* Reductions */

		MPI_Barrier(MPI_COMM_WORLD), gettimeofday(&start, NULL);
		double conv;
		for(int i = begin_x; i < end_x; i++){ 		//convergence

				for(int j = begin_y; j < end_y; j++){

						alpha =0.0;
						beta = 0.0;
						gamma = 0.0;
						for(int k = 0; k<N; k++){


								alpha = alpha + (U_t[P(i, k)] * U_t[P(i, k)]);
								beta = beta + (U_t[P(j ,k)] * U_t[P(j, k)]);
								gamma = gamma + (U_t[P(i, k)] * U_t[P(j, k)]);
						}
						Alphas[PP(i, j)] = alpha;
						Betas[PP(i, j)] = beta;
						Gammas[PP(i, j)] = gamma;

				}
		}

		int dis = dis_y * dis_x;
		MPI_Request requests[48];
		MPI_Status status;
		int req_count = 0;

		if (myrank == 0 ) {
				for (int i = 0; i < len_x; i ++ ) {
						for (int j = 0; j < len_y; j ++ ) {
								int pos = len_y * i + j; 
								if (pos != 0) {
										double *buffer_pos;  
										buffer_pos = Alphas + pos * dis;  
										MPI_Irecv(buffer_pos, dis, 
											MPI_DOUBLE, pos, 0, MPI_COMM_WORLD,
											&requests[req_count ++]);	
										buffer_pos = Betas + pos * dis;  
										MPI_Irecv(buffer_pos, dis, 
											MPI_DOUBLE, pos, 1, MPI_COMM_WORLD,
											&requests[req_count ++]);	
										buffer_pos = Gammas + pos * dis;  
										MPI_Irecv(buffer_pos, dis, 
											MPI_DOUBLE, pos, 2, MPI_COMM_WORLD,
											&requests[req_count ++]);	
								}
						}
				}
		}
		else {
				MPI_Isend(Alphas, dis, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requests[req_count++]);
				MPI_Isend(Betas, dis, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &requests[req_count++]);
				MPI_Isend(Gammas, dis, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &requests[req_count++]);
		}

		for (int i = 0; i < req_count; i ++) {
				MPI_Wait(&requests[i], &status);
		}

		MPI_Barrier(MPI_COMM_WORLD), gettimeofday(&end, NULL);

		// fix final result


		//Output time and iterations

		if(T=="-t" || P =="-t"){
				elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
				elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
				cout<<"Time: "<<elapsedTime<<" ms."<<endl<<endl;


		}


		// Output the matrixes for debug
		if(T== "-p" || P == "-p" && myrank == 0){
				cout<<"Alphas"<<endl<<endl;
				for(int i =0; i<M; i++){

						for(int j =0; j<N;j++){

								cout<<Alphas[PPP(i, j)]<<"  ";
						}
						cout<<endl;
				}

				cout<<endl<<"Betas"<<endl<<endl;
				for(int i =0; i<M; i++){

						for(int j=0; j<N;j++){	  
								cout<<Betas[PPP(i, j)]<<"  ";
						}
						cout<<endl;
				}

				cout<<endl<<"Gammas"<<endl<<endl;
				for(int i =0; i<M; i++){
						for(int j =0; j<N; j++){

								cout<<Gammas[PPP(i, j)]<<"  ";

						}
						cout<<endl;
				}

		}

		//Generate files for debug purpouse
		if((Db == "-d" || T == "-d" || P == "-d") && myrank == 0){


				ofstream Af;
				//file for Matrix A
				Af.open("AlphasMPI.mat"); 
				/*    Af<<"# Created from debug\n# name: A\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";*/

				Af<<M<<"  "<<N;
				for(int i = 0; i<M;i++){
						for(int j =0; j<M;j++){
								Af<<" "<<Alphas[PPP(i, j)];
						}
						Af<<"\n";
				}

				Af.close();

				ofstream Uf;

				//File for Matrix U
				Uf.open("BetasMPI.mat");
				/*    Uf<<"# Created from debug\n# name: Ugpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";*/

				for(int i = 0; i<M;i++){
						for(int j =0; j<M;j++){
								Uf<<" "<<Betas[PPP(i, j)];
						}
						Uf<<"\n";
				}
				Uf.close();

				ofstream Vf;
				//File for Matrix V
				Vf.open("GammasMPI.mat");
				/*    Vf<<"# Created from debug\n# name: Vgpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";*/

				for(int i = 0; i<M;i++){
						for(int j =0; j<M;j++){
								Vf<<" "<<Gammas[PPP(i, j)];
						}
						Vf<<"\n";
				}


				Vf.close();

				ofstream Sf;


		}

		/*
		for(int i = 0; i<M;i++){
				delete [] Alphas[i];
				delete [] U_t[i];
				delete [] Betas[i];
				delete [] Gammas[i];

		}
		delete [] Alphas;
		delete [] Betas;
		delete [] Gammas;
		delete [] U_t;
*/
		MPI_Finalize();
		return 0;
}
