import numpy as np
import itertools
from PIL import Image
import scipy.ndimage
import scipy.optimize
import scipy.stats
import glob
import types

image_directory = './pics'

show_solution = True

max_fovs = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

match_radius = .01

max_error = .01

downsample_factor = 2

filter_radius = 2

filter_width = filter_radius * 2 + 1

catalog_fill_factor = .5

num_catalog_bins = 25

max_stars_per_fov = 10

magnitude_minimum = 5.0

min_angle = .002

pattern_size = 5

min_pixels_in_group = 3

window_radius = 2

max_pattern_checking_stars = 8

max_mismatch_probability = 1e-20

fine_sky_map_fill_factor = .5

num_fine_sky_map_bins = 100

course_sky_map_fill_factor = .5

num_course_sky_map_bins = 4

avalanche_constant = 2654435761

STARN = 9110

bsc5_data_type = [("XNO", np.float32),
                  ("SRA0", np.float64),
                  ("SDEC0", np.float64),
                  ("IS", np.int16),
                  ("MAG", np.int16),
                  ("XRPM", np.float32),
                  ("XDPM", np.float32)
                  ]

bsc5_file = open('BSC5', 'rb')
bsc5_file.seek(28)
bsc5 = np.fromfile(bsc5_file, dtype=bsc5_data_type, count=STARN)

print(bsc5)

year = 2016

stars = []
for star_num in range(STARN):

  mag = bsc5[star_num][4] / 100.0

  if mag <= magnitude_minimum:
    ra = bsc5[star_num][1]
    ra += bsc5[star_num][5] * (year - 1950)
    dec = bsc5[star_num][2]
    dec += bsc5[star_num][6] * (year - 1950)
    print("ra", type(ra))

    if ra == 0.0 and dec == 0.0:
      continue

    vector = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])
    star_id = int(bsc5[star_num][0])
    stars.append((vector, mag, star_id))
  
def hash_code_to_index(hash_code, bins_per_dimension, hash_table_size):
  integer_hash_code = sum(int(hash_code[i]) * bins_per_dimension ** i for i in range(len(hash_code)))
  index = (integer_hash_code * avalanche_constant) % hash_table_size
  return index
  
def get_nearby_stars_compressed_course(vector, radius):
  nearby_star_ids = []
  # given error of at most radius in each dimension, compute the space of hash codes to lookup in the sky map
  hash_code_space = [range(max(low,0), min(high+1,2*num_course_sky_map_bins)) for (low, high) in zip(((vector + 1 - radius) * num_course_sky_map_bins).astype(np.int),
                                                                                                     ((vector + 1 + radius) * num_course_sky_map_bins).astype(np.int))]
  for hash_code in itertools.product(*hash_code_space):

    hash_index = hash_code_to_index(hash_code, 2*num_course_sky_map_bins, compressed_course_sky_map_hash_table_size)

    for index in ((2 * (hash_index + offset ** 2)) % compressed_course_sky_map_hash_table_size for offset in itertools.count()):

      if not compressed_course_sky_map[index]:
        break

      indices = compressed_course_sky_map[index:index+2]
      star_id_list = compressed_course_sky_map[slice(*indices)]
      first_star_vector = star_table[star_id_list[0]]
      first_star_hash_code = tuple(((first_star_vector+1)*num_course_sky_map_bins).astype(np.int))

      if first_star_hash_code == hash_code:
        for star_id in star_id_list:
          if np.dot(vector, star_table[star_id]) > np.cos(radius):
            nearby_star_ids.append(star_id)

  return nearby_star_ids

parameters = (max_fovs, 
              num_catalog_bins, 
              max_stars_per_fov, 
              magnitude_minimum, 
              min_angle, 
              pattern_size, 
              fine_sky_map_fill_factor, 
              num_fine_sky_map_bins,
              course_sky_map_fill_factor, 
              num_course_sky_map_bins)

try:
  pattern_catalog = np.load('pattern_catalog.npy')
  fine_sky_map = np.load('fine_sky_map.npy')
  compressed_course_sky_map = np.load('compressed_course_sky_map.npy')
  compressed_course_sky_map_hash_table_size = compressed_course_sky_map[-1]
  star_table = np.load('star_table.npy')
  stored_parameters = open('params.txt', 'r').read()
  read_failed = 0
except:
  read_failed = 1

if read_failed or str(parameters) != stored_parameters:

  STARN = 9110

  bsc5_data_type = [("XNO", np.float32),
                    ("SRA0", np.float64),
                    ("SDEC0", np.float64),
                    ("IS", np.int16),
                    ("MAG", np.int16),
                    ("XRPM", np.float32),
                    ("XDPM", np.float32)
                   ]
  
  bsc5_file = open('BSC5', 'rb')
  bsc5_file.seek(28)
  bsc5 = np.fromfile(bsc5_file, dtype=bsc5_data_type, count=STARN)

  print(bsc5)

  year = 2016

  stars = []
  for star_num in range(STARN):

    mag = bsc5[star_num][4] / 100.0

    if mag <= magnitude_minimum:
      ra = bsc5[star_num][1]
      ra += bsc5[star_num][5] * (year - 1950)
      dec = bsc5[star_num][2]
      dec += bsc5[star_num][6] * (year - 1950)

      if ra == 0.0 and dec == 0.0:
        continue

      vector = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])
      star_id = int(bsc5[star_num][0])
      stars.append((vector, mag, star_id))
        
  stars.sort(key=lambda star: star[0][0])
  doubles = [0] * len(stars)

  for star_num1 in range(len(stars)):
    for star_num2 in range(star_num1 + 1, len(stars)):

      if stars[star_num2][0][0] - stars[star_num1][0][0] >= min_angle:
        break

      if np.dot(stars[star_num1][0], stars[star_num2][0]) > np.cos(min_angle):
        doubles[star_num1] = 1
        doubles[star_num2] = 1
        break

  stars_no_doubles = [stars[i] for i in range(len(stars)) if not doubles[i]]

  print("number of stars in star table and sky maps: " + str(len(stars)))
  
  star_table = np.zeros((STARN+1, 3), dtype=np.float32)
  fine_sky_map = np.zeros(int(len(stars) / fine_sky_map_fill_factor), dtype=np.uint16)
  course_sky_map = {}

  for (vector, mag, star_id) in stars_no_doubles:

    star_table[star_id] = vector
    hash_code = ((vector+1)*num_fine_sky_map_bins).astype(np.int)
    hash_index = hash_code_to_index(hash_code, 2*num_fine_sky_map_bins, fine_sky_map.size)

    for index in ((hash_index + offset ** 2) % fine_sky_map.size for offset in itertools.count()):
      if not fine_sky_map[index]:
        fine_sky_map[index] = star_id
        break

    hash_code = tuple(((vector+1)*num_course_sky_map_bins).astype(np.int))
    course_sky_map[hash_code] = course_sky_map.pop(hash_code, []) + [star_id]

  compressed_course_sky_map_hash_table_size = 2 * len(course_sky_map.keys()) / course_sky_map_fill_factor
  compressed_course_sky_map = np.zeros(int(compressed_course_sky_map_hash_table_size + len(stars_no_doubles) + 1), dtype=np.uint16)
  compressed_course_sky_map[-1] = compressed_course_sky_map_hash_table_size
  first_open_slot_in_superlist = compressed_course_sky_map_hash_table_size

  for (hash_code, star_id_list) in course_sky_map.items():
    slice_indices = (first_open_slot_in_superlist, first_open_slot_in_superlist + len(star_id_list))
    slice_indices = tuple(map(int, slice_indices))
    compressed_course_sky_map[slice(*slice_indices)] = star_id_list
    first_open_slot_in_superlist += len(star_id_list)
    hash_index = hash_code_to_index(hash_code, 2*num_course_sky_map_bins, compressed_course_sky_map_hash_table_size)
    for index in ((2 * (hash_index + offset ** 2)) % compressed_course_sky_map_hash_table_size for offset in itertools.count()):
      index = int(index)
      if not compressed_course_sky_map[index]:
        compressed_course_sky_map[index:index+2] = slice_indices
        break
  
  stars_no_doubles.sort(key=lambda star: star[1])

  def get_nearby_stars_pruned_course(vector, radius):

    nearby_star_ids = []
    hash_code_space = [range(max(low,0), min(high+1,2*num_course_sky_map_bins)) for (low, high) in zip(((vector + 1 - radius) * num_course_sky_map_bins).astype(np.int),
                                                                                                       ((vector + 1 + radius) * num_course_sky_map_bins).astype(np.int))]

    for hash_code in itertools.product(*hash_code_space):
      for star_id in pruned_course_sky_map.get(hash_code, []):
        if np.dot(vector, star_table[star_id]) > np.cos(radius):
          nearby_star_ids.append(star_id)
    return nearby_star_ids
  
  print("generating catalog, this may take an hour...")
  pattern_list = np.zeros((100000000, pattern_size), dtype=np.uint16)
  num_patterns_found = 0

  for max_fov in max_fovs:

    print("computing " + str(max_fov) + " degree fov patterns...")
    max_fov_rad = max_fov * np.pi / 180
    pruned_course_sky_map = {}

    for (vector, mag, star_id) in stars_no_doubles:
      if len(get_nearby_stars_pruned_course(vector, max_fov_rad / 2)) >= max_stars_per_fov:
        continue
      hash_code = tuple(((vector+1)*num_course_sky_map_bins).astype(np.int))
      pruned_course_sky_map[hash_code] = pruned_course_sky_map.pop(hash_code, []) + [star_id]

    star_ids_pruned = [star_id for sublist in pruned_course_sky_map.values() for star_id in sublist]
    pattern = [None] * pattern_size

    for pattern[0] in star_ids_pruned:
      hash_code = tuple(np.floor((star_table[pattern[0]]+1)*num_course_sky_map_bins).astype(np.int))
      pruned_course_sky_map[hash_code].remove(pattern[0])
      for pattern[1:] in itertools.combinations(get_nearby_stars_pruned_course(star_table[pattern[0]], max_fov_rad), pattern_size-1):
        vectors = [star_table[star_id] for star_id in pattern]
        if all(np.dot(*star_pair) > np.cos(max_fov_rad) for star_pair in itertools.combinations(vectors[1:], 2)):
          pattern_list[num_patterns_found] = pattern
          num_patterns_found += 1

  pattern_list = pattern_list[:num_patterns_found]
  print("inserting patterns into catalog...")
  tmp = tuple(map(int, (num_patterns_found / catalog_fill_factor, pattern_size)))
  pattern_catalog = np.zeros(tmp, dtype=np.uint16)

  for pattern in pattern_list:
    vectors = np.array([star_table[star_id] for star_id in pattern])
    edges = np.sort([np.linalg.norm(np.subtract(*star_pair)) for star_pair in itertools.combinations(vectors, 2)])
    largest_edge = edges[-1]
    edge_ratios = edges[:-1] / largest_edge
    hash_code = tuple((edge_ratios * num_catalog_bins).astype(np.int))
    hash_index = hash_code_to_index(hash_code, num_catalog_bins, pattern_catalog.shape[0])
    for index in ((hash_index + offset ** 2) % pattern_catalog.shape[0] for offset in itertools.count()):
      index = int(index)
      if not pattern_catalog[index][0]:
        pattern_catalog[index] = pattern
        break
      elif sorted(pattern_catalog[index]) == sorted(pattern):
        break
      else:
        continue

  np.save('star_table.npy', star_table)
  np.save('fine_sky_map.npy', fine_sky_map)
  np.save('compressed_course_sky_map.npy', compressed_course_sky_map)
  np.save('pattern_catalog.npy', pattern_catalog)
  parameters = open('params.txt', 'w').write(str(parameters))
  
def tetra(image_file_name):

  image = np.array(Image.open(image_file_name).convert('L'))
  height, width = image.shape

  height = height - height % downsample_factor
  width = width - width % downsample_factor
  image = image[:height, :width]

  downsampled_image = image.reshape((height//downsample_factor,downsample_factor,width//downsample_factor,downsample_factor)).mean(axis=3).mean(axis=1)
  median_filtered_image = scipy.ndimage.filters.median_filter(downsampled_image, size=filter_width, output=image.dtype)
  upsampled_median_filtered_image = median_filtered_image.repeat(downsample_factor, axis=0).repeat(downsample_factor, axis=1)
  normalized_image = image - np.minimum.reduce([upsampled_median_filtered_image, image])

  bright_pixels = zip(*np.where(normalized_image > 5 * np.std(normalized_image)))
  pixel_to_group = {}

  for pixel in bright_pixels:
    left_pixel = (pixel[0]  , pixel[1]-1)
    up_pixel   = (pixel[0]-1, pixel[1]  )
    in_left_group = left_pixel in pixel_to_group
    in_up_group = up_pixel in pixel_to_group
    if in_left_group and in_up_group and id(pixel_to_group[left_pixel]) != id(pixel_to_group[up_pixel]):
      pixel_to_group[up_pixel].append(pixel)
      pixel_to_group[left_pixel].extend(pixel_to_group[up_pixel])
      for up_group_pixel in pixel_to_group[up_pixel]:
        pixel_to_group[up_group_pixel] = pixel_to_group[left_pixel]
    elif in_left_group:
      pixel_to_group[left_pixel].append(pixel)
      pixel_to_group[pixel] = pixel_to_group[left_pixel]
    elif in_up_group:
      pixel_to_group[up_pixel].append(pixel)
      pixel_to_group[pixel] = pixel_to_group[up_pixel]
    else:
      pixel_to_group[pixel] = [pixel]

  seen = set()
  groups = [seen.add(id(group)) or group for group in pixel_to_group.values() if id(group) not in seen]

  star_center_pixels = [max(group, key=lambda pixel: normalized_image[pixel]) for group in groups if len(group) > min_pixels_in_group]
  window_size = window_radius * 2 + 1
  x_weights = np.fromfunction(lambda y,x:x+.5,(window_size, window_size))
  y_weights = np.fromfunction(lambda y,x:y+.5,(window_size, window_size))
  star_centroids = []

  for (y,x) in star_center_pixels:
    if y < window_radius or y >= height - window_radius or \
       x < window_radius or x >= width  - window_radius:
      continue
    star_window = normalized_image[y-window_radius:y+window_radius+1, x-window_radius:x+window_radius+1]
    mass = np.sum(star_window)
    x_center = np.sum(star_window * x_weights) / mass - window_radius
    y_center = np.sum(star_window * y_weights) / mass - window_radius
    star_centroids.append((y + y_center, x + x_center))

  star_centroids.sort(key=lambda yx:-np.sum(normalized_image[int(yx[0]-window_radius):int(yx[0]+window_radius+1), int(yx[1]-window_radius):int(yx[1]+window_radius+1)]))

  def compute_vectors(star_centroids, fov):
    center_x = width / 2.
    center_y = height / 2.
    fov_rad = fov * np.pi / 180
    scale_factor = np.tan(fov_rad / 2) / center_x
    star_vectors = []
    for (star_y, star_x) in star_centroids:
      j_over_i = (center_x - star_x) * scale_factor
      k_over_i = (center_y - star_y) * scale_factor
      i = 1. / np.sqrt(1 + j_over_i**2 + k_over_i**2)
      j = j_over_i * i
      k = k_over_i * i
      star_vectors.append(np.array([i,j,k]))
    return star_vectors

  def centroid_pattern_generator(star_centroids, pattern_size):

    if len(star_centroids) < pattern_size:
      return

    star_centroids = np.array(star_centroids)
    pattern_indices = [-1] + list(range(pattern_size)) + [len(star_centroids)]
    yield star_centroids[pattern_indices[1:-1]]

    while pattern_indices[1] < len(star_centroids) - pattern_size:
      for index_to_change in range(1, pattern_size + 1):
        pattern_indices[index_to_change] += 1
        if pattern_indices[index_to_change] < pattern_indices[index_to_change + 1]:
          break
        else:
          pattern_indices[index_to_change] = pattern_indices[index_to_change - 1] + 1
      yield star_centroids[pattern_indices[1:-1]]
          
  for pattern_star_centroids in centroid_pattern_generator(star_centroids[:max_pattern_checking_stars], pattern_size):
    for fov_estimate in max_fovs:
      pattern_star_vectors = compute_vectors(pattern_star_centroids, fov_estimate)
      pattern_edges = np.sort([np.linalg.norm(np.subtract(*star_pair)) for star_pair in itertools.combinations(pattern_star_vectors, 2)])
      pattern_largest_edge = pattern_edges[-1]
      pattern_edge_ratios = pattern_edges[:-1] / pattern_largest_edge
      hash_code_space = [range(max(low,0), min(high+1,num_catalog_bins)) for (low, high) in zip(((pattern_edge_ratios - max_error) * num_catalog_bins).astype(np.int),
                                                                                                ((pattern_edge_ratios + max_error) * num_catalog_bins).astype(np.int))]

      for hash_code in set([tuple(sorted(code)) for code in itertools.product(*hash_code_space)]):

        hash_code = tuple(hash_code)
        hash_index = hash_code_to_index(hash_code, num_catalog_bins, pattern_catalog.shape[0])

        for index in ((hash_index + offset ** 2) % pattern_catalog.shape[0] for offset in itertools.count()):

          if not pattern_catalog[index][0]:
            break

          catalog_pattern = pattern_catalog[index]
          catalog_vectors = np.array([star_table[star_id] for star_id in catalog_pattern])
          centroid = np.mean(catalog_vectors, axis=0)
          radii = [np.linalg.norm(vector - centroid) for vector in catalog_vectors]
          catalog_sorted_vectors = catalog_vectors[np.argsort(radii)]
          catalog_edges = np.sort([np.linalg.norm(np.subtract(*star_pair)) for star_pair in itertools.combinations(catalog_vectors, 2)])
          catalog_largest_edge = catalog_edges[-1]
          catalog_edge_ratios = catalog_edges[:-1] / catalog_largest_edge

          if any([abs(val) > max_error for val in (catalog_edge_ratios - pattern_edge_ratios)]):
            continue

          catalog_edges = np.append(catalog_edge_ratios * catalog_largest_edge, catalog_largest_edge)

          def fov_to_error(fov):
            pattern_star_vectors = compute_vectors(pattern_star_centroids, fov)
            pattern_edges = np.sort([np.linalg.norm(np.subtract(*star_pair)) for star_pair in itertools.combinations(pattern_star_vectors, 2)])
            return catalog_edges - pattern_edges

          fov = scipy.optimize.leastsq(fov_to_error, fov_estimate)[0][0]
          fov_rad = fov * np.pi / 180
          fov_half_diagonal_rad = fov_rad * np.sqrt(width ** 2 + height ** 2) / (2 * width)
          pattern_star_vectors = compute_vectors(pattern_star_centroids, fov)
          pattern_centroid = np.mean(pattern_star_vectors, axis=0)
          pattern_radii = [np.linalg.norm(star_vector - pattern_centroid) for star_vector in pattern_star_vectors]
          pattern_sorted_vectors = np.array(pattern_star_vectors)[np.argsort(pattern_radii)]
          
          def find_rotation_matrix(image_vectors, catalog_vectors):
            H = np.sum([np.dot(image_vectors[i].reshape((3,1)), catalog_vectors[i].reshape((1,3))) for i in range(len(image_vectors))], axis=0)
            U, _, V = np.linalg.svd(H)
            rotation_matrix = np.dot(U, V)
            rotation_matrix[:,2] *= np.linalg.det(rotation_matrix)
            return rotation_matrix
          
          rotation_matrix = find_rotation_matrix(pattern_sorted_vectors, catalog_sorted_vectors)
          all_star_vectors = compute_vectors(star_centroids, fov)
          
          def find_matches(all_star_vectors, rotation_matrix):

            rotated_star_vectors = [np.dot(rotation_matrix.T, star_vector) for star_vector in all_star_vectors]
            catalog_vectors = []

            for rotated_star_vector in rotated_star_vectors:
              hash_code_space = [range(max(low,0), min(high+1,2*num_fine_sky_map_bins)) for (low, high) in zip(((rotated_star_vector + 1 - match_radius) * num_fine_sky_map_bins).astype(np.int),
                                                                                                               ((rotated_star_vector + 1 + match_radius) * num_fine_sky_map_bins).astype(np.int))]
              matching_stars = []

              for hash_code in [code for code in itertools.product(*hash_code_space)]:
                hash_index = hash_code_to_index(hash_code, 2*num_fine_sky_map_bins, fine_sky_map.size)
                for index in ((hash_index + offset ** 2) % fine_sky_map.size for offset in itertools.count()):
                  if not fine_sky_map[index]:
                    break
                  elif np.dot(star_table[fine_sky_map[index]], rotated_star_vector) > np.cos(match_radius * fov_rad):
                    matching_stars.append(star_table[fine_sky_map[index]])
              catalog_vectors.append(matching_stars)
              
            matches = [(image_vector, catalog_star[0]) for (image_vector, catalog_star) in zip(all_star_vectors, catalog_vectors) if len(catalog_star) == 1]
            # catalog stars must uniquely match image stars
            matches_hash = {}
            # add the matches to the hash one at a time
            for (image_vector, catalog_vector) in matches:
              # exactly one image vector must match
              if tuple(catalog_vector) in matches_hash:
                matches_hash[tuple(catalog_vector)] = "multiple matches"
              else:
                matches_hash[tuple(catalog_vector)] = image_vector
            # reverse order so that image vector is first in each pair
            matches = []
            for (catalog_vector, image_vector) in matches_hash.items():
              # filter out catalog stars with multiple image star matches
              if isinstance(image_vector, types.StringType) and image_vector == "multiple matches":
                continue
              matches.append((image_vector, np.array(catalog_vector)))
            return matches
          
          matches = find_matches(all_star_vectors, rotation_matrix)
          # calculate loose upper bound on probability of mismatch assuming random star distribution
          # find number of catalog stars appear in a circumscribed circle around the image
          image_center_vector = np.dot(rotation_matrix.T, np.array((1,0,0)))
          num_nearby_catalog_stars = len(get_nearby_stars_compressed_course(image_center_vector, fov_half_diagonal_rad))
          # calculate probability of a single random image centroid mismatching to a catalog star
          single_star_mismatch_probability = 1 - num_nearby_catalog_stars * match_radius ** 2 * width / height
          # apply binomial theorem to calculate probability upper bound on this many mismatches
          # three of the matches account for the dimensions of freedom: position, rotation, and scale
          mismatch_probability_upper_bound = scipy.stats.binom.cdf(len(star_centroids) - (len(matches) - 3), len(star_centroids) - 3, single_star_mismatch_probability)
          # if a high probability match has been found, recompute the attitude using all matching stars
          if mismatch_probability_upper_bound < max_mismatch_probability:
            # diplay mismatch probability in scientific notation
            print ("mismatch probability: %.4g" % mismatch_probability_upper_bound)
            # recalculate the rotation matrix using the newly identified stars
            rotation_matrix = find_rotation_matrix(*zip(*matches))
            # recalculate matched stars given new rotation matrix
            matches = find_matches(all_star_vectors, rotation_matrix)
            # extract right ascension, declination, and roll from rotation matrix and convert to degrees
            ra = (np.arctan2(rotation_matrix[0][1], rotation_matrix[0][0]) % (2 * np.pi)) * 180 / np.pi
            dec = np.arctan2(rotation_matrix[0][2], np.sqrt(rotation_matrix[1][2]**2 + rotation_matrix[2][2]**2)) * 180 / np.pi
            roll = (np.arctan2(rotation_matrix[1][2], rotation_matrix[2][2]) % (2 * np.pi)) * 180 / np.pi
            # print out attitude and field-of-view to 4 decimal places
            print("RA:   %.4f" % ra)
            print("DEC:  %.4f" % dec)
            print("ROLL: %.4f" % roll)
            print("FOV:  %.4f" % fov)
            # display input image with green circles around matched catalog stars
            # and red circles around unmatched catalog stars
            if show_solution:
              # draws circles around where the given vectors appear in an image
              def draw_circles(image, vectors, color, circle_fidelity, circle_radius):
                # calculate the pixel position of the center of the image
                image_center_x = width / 2.
                image_center_y = height / 2.
                # calculate conversion ratio between pixels and distance in the unit celestial sphere
                scale_factor = image_center_x / np.tan(fov_rad / 2)
                # iterate over the vectors, adding a circle for each one that appears in the image frame
                for (i, j, k) in vectors:
                  # find the center pixel for the vector's circle
                  circle_center_x = np.floor(image_center_x - (j / i) * scale_factor)
                  circle_center_y = np.floor(image_center_y - (k / i) * scale_factor)
                  # draw a circle of the given color with the given fidelity
                  for angle in np.array(range(circle_fidelity)) * 2 * np.pi / circle_fidelity:
                    # find the x and y coordinates for the pixel that will be drawn
                    pixel_x = int(circle_center_x + circle_radius * np.sin(angle))
                    pixel_y = int(circle_center_y + circle_radius * np.cos(angle))
                    # verify the pixel is within the image bounds
                    if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
                      continue
                    # draw the pixel
                    image.putpixel((pixel_x, pixel_y), color)
              # plot the image with green circles around matched stars
              # and red circles around stars that weren't matched
              rgb_image = Image.fromarray(image).convert('RGB')
              # the circle is drawn using the corners of an n-gon, where the circle fidelity is n
              circle_fidelity = 100
              # star centroids that appear within the circle radius would match with the circle's corresponding catalog vector
              circle_radius = match_radius * width + 1
              # find which catalog stars could appear in the image
              image_center_vector = np.dot(rotation_matrix.T, np.array((1,0,0)))
              nearby_catalog_stars = get_nearby_stars_compressed_course(image_center_vector, fov_half_diagonal_rad)
              # rotate the vectors of all of the nearby catalog stars into the image frame
              rotated_nearby_catalog_vectors = [np.dot(rotation_matrix, star_table[star_id]) for star_id in nearby_catalog_stars]
              # color all of the circles red by default
              color_all = (255, 0, 0)
              draw_circles(rgb_image, rotated_nearby_catalog_vectors, color_all, circle_fidelity, circle_radius)
              # rotate the matched catalog stars into the image frame
              matched_rotated_catalog_vectors = [np.dot(rotation_matrix, catalog_vector) for (image_vector, catalog_vector) in matches]
              # recolor matched circles green
              color_matched = (0, 255, 0)
              draw_circles(rgb_image, matched_rotated_catalog_vectors, color_matched, circle_fidelity, circle_radius)
              rgb_image.show()
            return

  # print failure message
  print("failed to determine attitude")

for image_file_name in glob.glob(image_directory + '/*'):
  print(image_file_name)
  tetra(image_file_name)
