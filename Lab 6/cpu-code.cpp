/*
 * Deepam Sarmah
 * 2020050
 * deepam20050@iiitd.ac.in
 */
#include <ctime>
#include <cstdio>

using namespace std;

const int ROWS = 1024;
const int COLS = 512;

float A[ROWS][COLS], A_T[COLS][ROWS], A_T_A[COLS][COLS];

void compute_transpose () {
  for (int i = 0; i < ROWS; ++i) {
    for (int j = 0; j < COLS; ++j) {
      A_T[j][i] = A[i][j];
    }
  }
}

void matrix_mult () {
  for (int i = 0; i < COLS; ++i) {
    for (int j = 0; j < COLS; ++j) {
      for (int k = 0; k < ROWS; ++k) {
        A_T_A[i][j] += A_T[i][k] * A[k][j];
      }
    }
  }
}

int main() {
  for (int i = 0; i < ROWS; ++i) {
    for (int j = 0; j < COLS; ++j) {
      A[i][j] = static_cast<float>((i + 1) * (j + 1));
    }
  }
  timespec begin, end;
  clock_gettime(CLOCK_REALTIME, &begin);
  compute_transpose();
  clock_gettime(CLOCK_REALTIME, &end);
  long seconds = end.tv_sec - begin.tv_sec, nanoseconds = end.tv_nsec - begin.tv_nsec;
  double elapsed = seconds + nanoseconds * 1e-9;
  printf("[CPU] Transpose Time measured: %.9lf seconds.\n", elapsed);
  clock_gettime(CLOCK_REALTIME, &begin);
  matrix_mult();
  clock_gettime(CLOCK_REALTIME, &end);
  seconds = end.tv_sec - begin.tv_sec;
  nanoseconds = end.tv_nsec - begin.tv_nsec;
  elapsed = seconds + nanoseconds * 1e-9;
  printf("[CPU] A_T * A Time measured: %.9lf seconds.\n", elapsed);
  return 0;
}
