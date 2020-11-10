#include <omp.h>
#include <iostream>

using namespace std;

int main(int argc, char* argv[]) {
    //Инициализация

    const int N = 1024; //размерность матриц
    int num_threads = 1;// число потоков, равное числу лог. ядер
    omp_set_dynamic(0);//разрешить изменение числа потоков
    omp_set_num_threads(num_threads); //установить число потоков
    int sch_block;//статические шаги цикла в клаузе schedule
    cout << "Input size of schedule block: ";
    cin >> sch_block;

    double time1, time2;
    int i, j, r, rank, size;
    auto** A = new float*[N];
    auto** B = new float*[N];
    auto** C = new float*[N];
    auto** Y = new float*[N];
    auto** T1 = new float*[N];
    auto** T2 = new float*[N];
    auto** T3 = new float*[N];
    for (i = 0; i < N; i++) {
        A[i] = new float[N];
        B[i] = new float[N];
        C[i] = new float[N];
        Y[i] = new float[N];
        T1[i] = new float[N];
        T2[i] = new float[N];
        T3[i] = new float[N];
    }
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = 1;
            B[i][j] = 2;
            C[i][j] = 3;
            //Y[i][j] = 0;
        }
    time1 = omp_get_wtime();

    //Y[i][j] = A[i][j] / C[i][j] + B[i][j] * (A[i][j] + C[i][j]);
    //T2 = A + C
#pragma omp parallel for private(j) schedule (static, sch_block)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            T2[i][j] = A[i][j] + C[i][j];
        }
    }

    // неявная барьерная синхронизация

    //Т1 = А/С
    //Т3 = B*(A+C)=B*T2
    //Y=A/C+B*(A+C)=T1+T3
#pragma omp parallel for private(j, r) schedule (static, sch_block)//schedule(static,N/4)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            T1[i][j] = A[i][j] / C[i][j];
            T3[i][j] = 0;
            for (r = 0; r < N;r++) {
                T3[i][j] += B[i][r] * T2[r][j];
            }
            Y[i][j] = T1[i][j] + T3[i][j];
        }
    }

    time2 = omp_get_wtime();
    cout << "Time: " << time2 - time1 << endl;
    cout << "T1[0][0] = " << T1[0][0] << endl;
    cout << "T2[0][0] = " << T2[0][0] << endl;
    cout << "T3[0][0] = " << T3[0][0] << endl;
    cout << "Y[0][0] = " << Y[0][0] << endl;

    //Освобождение памяти
    for (i = 0; i < N; i++) {
        delete[]A[i];
        delete[]B[i];
        delete[]C[i];
        delete[]Y[i];
        delete[]T1[i];
        delete[]T2[i];
        delete[]T3[i];
    }
    delete[]A;
    delete[]B;
    delete[]C;
    delete[]Y;
    delete[]T1;
    delete[]T2;
    delete[]T3;
    return 0;
}
