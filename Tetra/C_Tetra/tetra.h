#ifndef TETRA_H
#define TETRA_H

#define _FILE_OFFSET_BITS 64

#define NUM_STARS 5904

#define PATTERN_CACHE_SIZE 16

#define MAX_PROBE_DEPTH 4278

#define NUM_STARS_IN_PATTERN 4

#define CATALOG_SIZE_IN_PATTERNS 770708495

#define MAX_FOV 0.247

#define BIN_SIZE_RATIO 3.0

#define MAX_CENTROID_ERROR .00069054

#define MAX_FOV_ERROR 0.01

#define MAX_STARS 12

#define MAX_STARS_PER_IMAGE 25

#define NUM_IMAGES 1000

#define PI 3.1415926

#define MAX_SCALE_FACTOR (fmax(tan(MAX_FOV*(1+MAX_FOV_ERROR)/2.0)/tan(MAX_FOV/2.0),1-tan(MAX_FOV*(1-MAX_FOV_ERROR)/2.0)/tan(MAX_FOV/2.0)))

#define LE_ERROR_SLOPE (MAX_SCALE_FACTOR-1)

#define LE_ERROR_OFFSET (2*MAX_CENTROID_ERROR/(2-MAX_SCALE_FACTOR))

#define MAX_LE_LENGTH (2*sin(MAX_FOV*(1+MAX_FOV_ERROR)/2.0))

typedef struct feature Feature;
typedef struct pattern Pattern;
typedef struct star Star;
typedef struct param Param;

#endif