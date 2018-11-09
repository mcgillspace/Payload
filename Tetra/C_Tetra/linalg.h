#ifndef LINALG_H
#define LINALG_H

void diff(double v1[3], double v2[3], double output[3]);
double mag(double v[3]);
void normalize(double v[3]);
double dist(double v1[3], double v2[3]);
double dot_prod(double v1[3], double v2[3]);
void cross_prod(double v1[3], double v2[3], double output[3]);

#endif