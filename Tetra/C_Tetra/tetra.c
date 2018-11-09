#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "linalg.h"
#include "tetra.h"

struct param
{
  double ra;
  double dec;
  double roll;
  double fov;
  double mismatch_prob;
};

struct feature
{
  int x : 15;
  unsigned int x_bin_offset : 1;
  int y : 15;
  unsigned int y_bin_offset : 1;
  unsigned int star_id : 15;
  unsigned int pad : 1;
};

struct pattern
{
  Feature features[NUM_STARS_IN_PATTERN-2];
  uint16_t largest_edge;
  #define has_pattern largest_edge
  unsigned int le_bin_offset : 1;
  unsigned int fixed_star_id_1 : 15;
  unsigned int fixed_star_id_2 : 15;
  unsigned int is_last : 1;
};

struct star
{
  double vec[3];
  double mag;
  unsigned int star_id;
};


static double get_base(double error_slope, double error_offset)
{
  if(error_offset <= 0)
  {
    perror("Non-positive error value detected: increase error values!\n");
    exit(EXIT_FAILURE);
  }

  return (1 + error_slope) / fmax(1-error_slope, 0);
}

static int log_bin(double input, double error_slope, double error_offset)
{
  if(!isfinite(error_slope) || !isfinite(error_offset))
    return 0;

  int bin;
  double base = get_base(error_slope, error_offset);
  if(base <= 1 + error_offset * BIN_SIZE_RATIO / 10.0)
  {
    bin = input / (2 * (error_slope + error_offset) * BIN_SIZE_RATIO);
  }
  else
  {
    bin = (log(input * error_slope / error_offset + 1) / log(base)) / BIN_SIZE_RATIO;
  }

  return bin;
}

static double log_unbin(int bin, double error_slope, double error_offset)
{
  if(!isfinite(error_slope) || !isfinite(error_offset))
    return 0;

  double min_input;
  double base = get_base(error_slope, error_offset);
  if(base <= 1 + error_offset * BIN_SIZE_RATIO / 10.0)
  {
    min_input = bin * 2 * (error_slope + error_offset) * BIN_SIZE_RATIO;
  }
  else
  {
    min_input = (pow(base, bin*BIN_SIZE_RATIO) - 1) * error_offset / error_slope;
  }

  return min_input;
}

static int bin_largest_edge(unsigned int largest_edge, int error_ratio)
{
  double le_ratio = largest_edge / ((1 << 16) - 1.0);
  le_ratio += error_ratio * (le_ratio * LE_ERROR_SLOPE + LE_ERROR_OFFSET);
  return log_bin(le_ratio, LE_ERROR_SLOPE, LE_ERROR_OFFSET);
}

static double unbin_largest_edge(unsigned int bin)
{
  return log_unbin(bin, LE_ERROR_SLOPE, LE_ERROR_OFFSET);
}

static int bin_y(int y, unsigned int le_bin, int error_ratio)
{
  double min_le_ratio = unbin_largest_edge(le_bin);
  double error_constant = LE_ERROR_OFFSET / (2 - MAX_SCALE_FACTOR);
  double error_slope = error_constant / fmax(min_le_ratio-error_constant, 0);
  double error_offset = error_slope;
  double y_ratio = y / ((1 << 14) - 1.0);
  y_ratio += error_ratio * copysign(fabs(y_ratio)*error_slope+error_offset, y_ratio);
  int bin = log_bin(fabs(y_ratio), error_slope, error_offset);

  return (y_ratio < 0) ? ~bin : bin;
}

static double unbin_y(int bin, unsigned int le_bin)
{
  double min_le_ratio = unbin_largest_edge(le_bin);
  double error_constant = LE_ERROR_OFFSET / (2 - MAX_SCALE_FACTOR);
  double error_slope = error_constant / fmax(min_le_ratio-error_constant, 0);
  double error_offset = error_slope;
  return log_unbin((bin >= 0) ? bin+1 :(~bin)+1, error_slope, error_offset);
}

static int bin_x(int x, unsigned int le_bin, int y_bin, int error_ratio)
{
  double min_le_ratio = unbin_largest_edge(le_bin);
  double max_y_ratio = unbin_y(y_bin, le_bin);
  double error_constant = LE_ERROR_OFFSET/(2-MAX_SCALE_FACTOR);
  double error_slope = error_constant/fmax(min_le_ratio-error_constant, 0);
  double error_offset = error_slope*(1+2*sqrt((1.0/4)+max_y_ratio*max_y_ratio))/2;
  double x_ratio = x/((1<<14)-1.0);
  x_ratio += error_ratio * copysign(fabs(x_ratio)*error_slope+error_offset, x_ratio);
  int bin = log_bin(fabs(x_ratio), error_slope, error_offset);

  return (x_ratio < 0) ? ~bin : bin;
}

static uint64_t hash_int(uint64_t old_hash, uint64_t key)
{
  key *= 11400714819323198549ULL;
  return old_hash ^ (old_hash >> 13) ^ key;
}

static uint64_t hash_pattern(Pattern pattern_instance)
{
  unsigned int le_bin = bin_largest_edge(pattern_instance.largest_edge, 0);
  uint64_t hash = hash_int(0, le_bin);

  for(int i = 0; i < NUM_STARS_IN_PATTERN-2; i++)
  {
    int y_bin = bin_y(pattern_instance.features[i].y, le_bin, 0);
    hash = hash_int(hash, y_bin+(1<<31));
    hash = hash_int(hash, bin_x(pattern_instance.features[i].x, le_bin, y_bin, 0)+(1<<31));
  }

  return hash % CATALOG_SIZE_IN_PATTERNS;
}

static int hash_same(Pattern new_pattern, Pattern cat_pattern)
{
  unsigned int new_le_bin = bin_largest_edge(new_pattern.largest_edge, 0);
  unsigned int cat_le_bin = bin_largest_edge(cat_pattern.largest_edge, 2*cat_pattern.le_bin_offset-1);
  if(new_le_bin != cat_le_bin)
    return 0;

  for(int i = 0; i < NUM_STARS_IN_PATTERN-2; i++)
  {
    Feature new_feature = new_pattern.features[i];
    Feature cat_feature = cat_pattern.features[i];

    int new_y_bin = bin_y(new_feature.y, new_le_bin, 0);
    int cat_y_bin = bin_y(cat_feature.y, cat_le_bin, 2*cat_feature.y_bin_offset-1);
    int new_x_bin = bin_x(new_feature.x, new_le_bin, new_y_bin, 0);
    int cat_x_bin = bin_x(cat_feature.x, cat_le_bin, cat_y_bin, 2*cat_feature.x_bin_offset-1);

    if((new_y_bin != cat_y_bin) || (new_x_bin != cat_x_bin))
      return 0;
  }

  return 1;
}

static int is_match(Pattern new_pattern, Pattern cat_pattern)
{
  double new_le_ratio = new_pattern.largest_edge / ((1 << 16) - 1.0);
  double cat_le_ratio = cat_pattern.largest_edge / ((1 << 16) - 1.0);
  double max_le_error = cat_le_ratio * LE_ERROR_SLOPE + LE_ERROR_OFFSET;

  if(fabs(new_le_ratio - cat_le_ratio) > max_le_error)
    return 0;

  double coord_error_constant = LE_ERROR_OFFSET / (2 - MAX_SCALE_FACTOR);
  double coord_error_slope = coord_error_constant / fmax(new_le_ratio-coord_error_constant, 0);
  double coord_error_offset_y = coord_error_slope;

  for(int i = 0; i < NUM_STARS_IN_PATTERN-2; i++)
  {
    double new_y = new_pattern.features[i].y / ((1 << 14 )- 1.0);
    double cat_y = cat_pattern.features[i].y / ((1 << 14) - 1.0);
    double max_y_error = fabs(cat_y) * coord_error_slope + coord_error_offset_y;

    if(fabs(new_y-cat_y) > max_y_error)
      return 0;
  }

  unsigned int cat_le_bin = bin_largest_edge(cat_pattern.largest_edge, 2*cat_pattern.le_bin_offset-1);

  for(int i = 0; i < NUM_STARS_IN_PATTERN-2; i++)
  {
    int cat_y_bin = bin_y(cat_pattern.features[i].y, cat_le_bin, 2*cat_pattern.features[i].y_bin_offset-1);
    double max_y_ratio = unbin_y(cat_y_bin, cat_le_bin);
    double coord_error_offset_x = coord_error_slope * (1 + 2 * sqrt((1.0 / 4) + max_y_ratio * max_y_ratio)) / 2;
    double new_x = new_pattern.features[i].x / ((1 << 14) - 1.0);
    double cat_x = cat_pattern.features[i].x / ((1 << 14) - 1.0);
    double max_x_error = fabs(cat_x) * coord_error_slope + coord_error_offset_x;

    if(fabs(new_x-cat_x) > max_x_error)
      return 0;
  }

  return 1;
}

static int increment_offset(FILE *pattern_catalog, Pattern catalog_pattern_cache[PATTERN_CACHE_SIZE], uint64_t *offset, int *cache_offset, int *probe_step)
{
  if(((*probe_step) * (*probe_step + 1)) / 2 > MAX_PROBE_DEPTH)
    return 0;

  *cache_offset += *probe_step;

  if (*cache_offset >= PATTERN_CACHE_SIZE)
  {
    *offset += *cache_offset;
    *cache_offset = 0;
    // _fseeki64(pattern_catalog, (*offset) * sizeof(Pattern), SEEK_SET);
    fseeko(pattern_catalog, (*offset)*sizeof(Pattern), SEEK_SET);
    fread(catalog_pattern_cache, sizeof(Pattern), PATTERN_CACHE_SIZE, pattern_catalog);
  }

  *probe_step += 1;

  return 1;
}

static int get_matching_pattern(Pattern image_pattern, Pattern *catalog_pattern, FILE *pattern_catalog)
{
  static Pattern catalog_pattern_cache[PATTERN_CACHE_SIZE];
  
  int cache_offset = 0;
  int probe_step = 1;
  int found_match = 0;
  uint64_t offset = hash_pattern(image_pattern);
  // _fseeki64(pattern_catalog, offset*sizeof(Pattern), SEEK_SET);
  fseeko(pattern_catalog, offset*sizeof(Pattern), SEEK_SET);
  fread(catalog_pattern_cache, sizeof(Pattern), PATTERN_CACHE_SIZE, pattern_catalog);

  while(catalog_pattern_cache[cache_offset].has_pattern)
  {
    if(hash_same(image_pattern, catalog_pattern_cache[cache_offset]))
    {
      if(is_match(image_pattern, catalog_pattern_cache[cache_offset]))
      {
        if(found_match)
          return 0;

        *catalog_pattern = catalog_pattern_cache[cache_offset];
        found_match = 1;
      }

      if(catalog_pattern_cache[cache_offset].is_last)
        break;
    }

    if(!increment_offset(pattern_catalog, catalog_pattern_cache, &offset, &cache_offset, &probe_step))
      return 0;
  }

  if(found_match)
    return 1;
  
  return 0;
}

static int identify_stars(double image_stars[MAX_STARS][3], int image_star_ids[NUM_STARS_IN_PATTERN], FILE *pattern_catalog, int matches[NUM_STARS_IN_PATTERN][2])
{
  int i,j;
	Pattern new_pattern;
	Pattern catalog_pattern;
  double largest_edge_length = 0.0;

  for(i = 0; i < NUM_STARS_IN_PATTERN; i++)
  {
    for(j = i+1; j< NUM_STARS_IN_PATTERN; j++)
    {
      double new_edge_length = dist(image_stars[image_star_ids[i]], image_stars[image_star_ids[j]]);
      if(new_edge_length > largest_edge_length)
      {
        largest_edge_length = new_edge_length;
        new_pattern.fixed_star_id_1 = image_star_ids[i];
        new_pattern.fixed_star_id_2 = image_star_ids[j];
      }
    }
  }
  
  new_pattern.largest_edge = (largest_edge_length / MAX_LE_LENGTH) * ((1 << 16) - 1);

  double x_axis_vector[3];
  diff(image_stars[new_pattern.fixed_star_id_2], image_stars[new_pattern.fixed_star_id_1], x_axis_vector);

  double y_axis_vector[3];
  cross_prod(image_stars[new_pattern.fixed_star_id_2], image_stars[new_pattern.fixed_star_id_1], y_axis_vector);

  normalize(x_axis_vector);
  normalize(y_axis_vector);

  int feature_index = 0;
  for(i = 0; i < NUM_STARS_IN_PATTERN; i++)
  {
    if(image_star_ids[i] != new_pattern.fixed_star_id_1 && image_star_ids[i] != new_pattern.fixed_star_id_2)
    {
      new_pattern.features[feature_index].star_id = image_star_ids[i];
      double x = dot_prod(x_axis_vector, image_stars[image_star_ids[i]]) / largest_edge_length;
      double y = dot_prod(y_axis_vector, image_stars[image_star_ids[i]]) / largest_edge_length;
      new_pattern.features[feature_index].x = x * ((1 << 14) - 1);
      new_pattern.features[feature_index].y = y * ((1 << 14) - 1);

      if(new_pattern.features[feature_index].x == 0)
        new_pattern.features[feature_index].x = 1;

      if(new_pattern.features[feature_index].y == 0)
        new_pattern.features[feature_index].y = 1;

      feature_index++;
    }
  }
  
  int pattern_rotation;
  unsigned int le_bin = bin_largest_edge(new_pattern.largest_edge, 0);
  int compare_bins(const void *p, const void *q) 
  {
    Feature *p_feature = (Feature*)p;
    Feature *q_feature = (Feature*)q;

    int p_y_bin = bin_y(p_feature->y, le_bin, 0);
    int q_y_bin = bin_y(q_feature->y, le_bin, 0);
    int p_x_bin = bin_x(p_feature->x, le_bin, p_y_bin, 0);
    int q_x_bin = bin_x(q_feature->x, le_bin, q_y_bin, 0);

    return (p_x_bin != q_x_bin) ? p_x_bin-q_x_bin : p_y_bin-q_y_bin;
  }

  qsort(new_pattern.features, NUM_STARS_IN_PATTERN-2, sizeof(Feature), compare_bins);

  Feature first_feature = new_pattern.features[0];
  first_feature.x = -first_feature.x;
  first_feature.y = -first_feature.y;
  pattern_rotation = compare_bins((void*)&first_feature, (void*)&(new_pattern.features[NUM_STARS_IN_PATTERN-3]));
  
  if(pattern_rotation >= 0)
  {
    for(i = 0; i < NUM_STARS_IN_PATTERN-2; i++)
    {
      new_pattern.features[i].x = -new_pattern.features[i].x;
      new_pattern.features[i].y = -new_pattern.features[i].y;
    }

    for(i = 0; i < (NUM_STARS_IN_PATTERN-2)/2; i++)
    {
      Feature feature_swap = new_pattern.features[i];
      new_pattern.features[i] = new_pattern.features[NUM_STARS_IN_PATTERN-3-i];
      new_pattern.features[NUM_STARS_IN_PATTERN-3-i] = feature_swap;
    }

    unsigned int fixed_star_id_swap = new_pattern.fixed_star_id_1;
    new_pattern.fixed_star_id_1 = new_pattern.fixed_star_id_2;
    new_pattern.fixed_star_id_2 = fixed_star_id_swap;
  }

	if(!get_matching_pattern(new_pattern, &catalog_pattern, pattern_catalog))
		return 0;

  matches[0][0] = new_pattern.fixed_star_id_1;
  matches[1][0] = new_pattern.fixed_star_id_2;
  matches[0][1] = catalog_pattern.fixed_star_id_1;
  matches[1][1] = catalog_pattern.fixed_star_id_2;

  for(i = 0; i < NUM_STARS_IN_PATTERN-2; i++)
  {
    matches[i+2][0] = new_pattern.features[i].star_id;
    matches[i+2][1] = catalog_pattern.features[i].star_id;
  }

	return 1;
}

static int identify_image(double image_stars[MAX_STARS][3], FILE *pattern_catalog, int num_image_stars, int matches[NUM_STARS_IN_PATTERN][2], int num_stars_selected)
{
  static int image_star_ids[NUM_STARS_IN_PATTERN];

  if(num_stars_selected < NUM_STARS_IN_PATTERN)
  {
    for(image_star_ids[num_stars_selected] = NUM_STARS_IN_PATTERN-num_stars_selected-1; image_star_ids[num_stars_selected] < num_image_stars; image_star_ids[num_stars_selected]++)
    {
      if(identify_image(image_stars, pattern_catalog, image_star_ids[num_stars_selected], matches, num_stars_selected+1))
      {
        return 1;
      }
    }
  }
  else
  {
    if(identify_stars(image_stars, image_star_ids, pattern_catalog, matches))
      return 1;
  }

  return 0;
}

int main(int argc, char *argv[]) 
{

	clock_t begin, end;
	int ms_spent;

	FILE *pattern_catalog;
  int matches[NUM_STARS_IN_PATTERN][2];

	int num_pixels_x = 1024;
	int num_pixels_y = 1024;
  
	pattern_catalog = fopen("pattern_catalog","rb");
	if(!pattern_catalog)
  {
		perror("Unable to open pattern catalog file!\n");
		return 1;
	}

  FILE *centroid_data;
	centroid_data = fopen("centroid_data.p","rb");
	if(!centroid_data)
  {
		perror("Unable to open centroid file!");
		return 1;
	}

	uint16_t *image_data = (uint16_t *)malloc(sizeof(uint16_t)*NUM_IMAGES*MAX_STARS_PER_IMAGE);
	FILE *image_data_file = fopen("image_data.p", "rb");
	if(!image_data_file)
  {
		perror("Unable to open image file!\n");
		return 1;
	}

  printf("%d\n", fileno(image_data_file));

	fread(image_data, sizeof(uint16_t), NUM_IMAGES*MAX_STARS_PER_IMAGE, image_data_file);	
	fclose(image_data_file);

	for(double fov = 10.1; fov < 10.1001; fov += .05) 
  {
    double fov_factor = (tan(fov*PI/360) * 2) / num_pixels_x;
    for(int centroid_error = 100; centroid_error < 101; centroid_error += 10) 
    {
      begin = clock();
      int right = 0;
      int failed = 0;
      int wrong = 0;
      int wrong_order = 0;
      int too_few_stars = 0;
      srand(0);

      int count = 0;
      for (int file_index = 0; file_index < NUM_IMAGES; file_index++) 
      {
        int i;
        int num_image_stars;
        float centroids[MAX_STARS][2];

        fseek(centroid_data, sizeof(float)*2*MAX_STARS_PER_IMAGE*file_index, SEEK_SET);
        fread(centroids, sizeof(float), MAX_STARS*2, centroid_data);

        for (num_image_stars = 0; num_image_stars < MAX_STARS; num_image_stars++) 
        {
          if (centroids[num_image_stars][0] == 0 && centroids[num_image_stars][1] == 0) 
            break;
        }

        if (num_image_stars < NUM_STARS_IN_PATTERN) 
        {
          too_few_stars++;
          continue;
        }
        
        double image_stars[MAX_STARS][3] = {{0}};

        double x;
        double y;
        for (i = 0; i < num_image_stars; i++) 
        {
          float rand1 = (float)((rand() % 10000) + 1) / 10000;
          float rand2 = (float)((rand() % 10000) + 1) / 10000;
          count++;
          float x_err = 2 * (rand1 - 0.5);
          float y_err = 2 * (rand2 - 0.5);

          while (sqrt(x_err*x_err + y_err*y_err) > 1.0) 
          {
            rand1 = (float)((rand() % 10000) + 1) / 10000;
            rand2 = (float)((rand() % 10000) + 1) / 10000;
            count++;
            x_err = 2 * (rand1 - 0.5);
            y_err = 2 * (rand2 - 0.5);
          }

          x = (centroids[i][0] + (x_err * centroid_error / 100)) * fov_factor;
          y = (centroids[i][1] + (y_err * centroid_error / 100)) * fov_factor;
          image_stars[i][0] = 1 / sqrt(1 + x*x + y*y);
          image_stars[i][1] = -image_stars[i][0] * x;
          image_stars[i][2] = image_stars[i][0] * y;
        }

        if(identify_image(image_stars, pattern_catalog, num_image_stars, matches, 0))
        {
          int num_verified_stars = 0;
          for (int v = 0; v < MAX_STARS_PER_IMAGE; v++) 
          {
            uint16_t image_data_star = image_data[file_index*MAX_STARS_PER_IMAGE + v];
            for(int match_id = 0; match_id < NUM_STARS_IN_PATTERN; match_id++)
            {
              if(matches[match_id][1] == image_data_star)
              {
                num_verified_stars++;
                break;
              }
            }
          }

          if (num_verified_stars < NUM_STARS_IN_PATTERN) 
          {
            wrong++;
          }
          else 
          {
            int mismatch = 0;
            for(int match_id = 0; match_id < NUM_STARS_IN_PATTERN; match_id++)
            {
              if(matches[match_id][1] != image_data[file_index*MAX_STARS_PER_IMAGE + matches[match_id][0]])
              {
                mismatch = 1;
                break;
              }
            }

            if(mismatch)
            {
              wrong_order++;
            }              
            else
            {
              right++; 
            }                        
          }							
        }
        else
        {
          failed++;
        }
      }

      end = clock();
      ms_spent = (int) 1000 * (end - begin) / CLOCKS_PER_SEC;

      printf("fov: %.2f\n", roundf(fov * 100) / 100);
      printf("centroid_error: %.2f\n", ((float) centroid_error / 100));
      printf("number right: %d\n", right);
      printf("number failed: %d\n", failed);
      printf("num wrngordr: %d\n", wrong_order);
      printf("number wrong: %d\n", wrong);
      printf("num 2 few stars: %d\n", too_few_stars);
      printf("ms taken: %d\n", ms_spent);

      FILE *output_file = fopen("output.txt", "a");
      fprintf(output_file, "fov: %.2f\n", roundf(fov * 100) / 100);
      fprintf(output_file, "centroid_error: %.2f\n", ((float) centroid_error / 100));
      fprintf(output_file, "number right: %d\n", right);
      fprintf(output_file, "number failed: %d\n", failed);
      fprintf(output_file, "num wrngordr: %d\n", wrong_order);
      fprintf(output_file, "number wrong: %d\n", wrong);
      fprintf(output_file, "ms taken: %d\n", ms_spent);
      fclose(output_file);
    }
  }

  fclose(pattern_catalog);	
  fclose(centroid_data);
}
