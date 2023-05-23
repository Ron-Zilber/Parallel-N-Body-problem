/// ------------------------ Parallel Computing - Assignment 2 ------------------------------------------------------ //
/// Authors: @ Shalom Yehuda Ben Yair @ Ron Zilbershtein ------------------------------------------------------------ //
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <math.h>

#define TOTAL_STARS_AMOUNT 992
#define TWO_MINUTES_RUNTIME 1500
#define KILO 1000
#define STAR_MASS 2e30
#define G_FORCE 6.674e-11
#define AVG_SPEED (200 * KILO)
#define LIGHT_YEAR (9e12 * KILO)
#define X 0
#define Y 1
#define RAND_NUMS_PER_STAR 4
#define UNIVERSE_BOUND (100 * LIGHT_YEAR)
#define TIME_STEP (2*10e9)

//  All physical constants are given in SI units
/// ------------------------ Function Declarations ------------------------------------------------------------------ //
typedef struct star {
    ///  Star structure that represents the bodies in the galaxy in terms of
    ///  ID, location, acceleration and velocity
    int ID[2];                //   ID[0] is the rank of the creating process. ID[2] is the index in the local stars list
    double location[2];                                                                    // location in form of (x, y)
    double acceleration[2];                                                            // acceleration in form of (x, y)
    double velocity[2];                                                                    // velocity in form of (x, y)
} star;

MPI_Datatype create_MPI_star();

double distance(const double *, const double *);

void update_acceleration(struct star *, const star *, int);

void update_velocity(struct star *, double);

void update_location(struct star *, double);

void initialize_stars(star *, int, int, const double *);

double *create_rand_nums(int);


/// ------------------------ main() --------------------------------------------------------------------------------- //
int main(int argc, char **argv) {
    ///  The code implements a parallel computation of the n-body problem,
    ///  which simulates the gravitational interactions among a system of celestial bodies.
    ///  The n-body problem calculates the positions and velocities of multiple bodies over time,
    ///  considering the gravitational forces acting between them.

    srand(time(NULL)); //  NOLINT(cert-msc30-c, cert-msc51-cpp) this comment shuts down a compilation warning
    //  from the IDE that claims srand's randomness is limited

    int N_stars = TOTAL_STARS_AMOUNT;                                             //  Set the number of all galaxy stars
    int numprocs, myID, i, local_list_length;
    int curr_stage = 0, total_stages = TWO_MINUTES_RUNTIME;
    double startwtime, endwtime, time_step = TIME_STEP;

    MPI_Init(&argc, &argv);                                                          //  Initialize that MPI environment
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myID);
    local_list_length = N_stars / numprocs;                 //  Each process will manage a group of This amount of stars

    //  Allocate memory for the local and the global stars lists
    star *local_stars_list = (star *) malloc(sizeof(star) * local_list_length);
    star *global_stars_list = (star *) malloc(sizeof(star) * N_stars);
    //  Create files for data exporting (each file contains a list of all stars locations at start, middle or end time

    FILE *start_file = NULL;
    FILE *mid_file = NULL;
    FILE *end_file = NULL;
    double *rand_nums;

    if (myID == 0) {                                                             //  Let the main process open the files
        start_file = fopen("start_file.csv", "w");
        mid_file = fopen("mid_file.csv", "w");
        end_file = fopen("end_file.csv", "w");
        //  Let the managing process generate random numbers (Four random number for each star)
        rand_nums = create_rand_nums(RAND_NUMS_PER_STAR * N_stars);
        startwtime = MPI_Wtime();
    }

    //  For each process, create a buffer that will hold a subset of the entire array
    double *local_rand_nums = (double *) malloc(sizeof(double) * RAND_NUMS_PER_STAR * local_list_length);

    //  Scatter the random numbers from the root process to all processes in the MPI world
    MPI_Scatter(rand_nums, RAND_NUMS_PER_STAR * local_list_length, MPI_DOUBLE, local_rand_nums,
                RAND_NUMS_PER_STAR * local_list_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //  Initialize N/P stars with random location and velocity
    initialize_stars(local_stars_list, local_list_length, myID, local_rand_nums);

    MPI_Datatype MPI_Star = create_MPI_star();  // Define an MPI datatype to assign as an argument for MPI_Allgather()
    while (curr_stage < total_stages) {
        //  Gather all lists from all processes into global_list:
        MPI_Allgather(local_stars_list, local_list_length, MPI_Star, global_stars_list,
                      local_list_length,
                      MPI_Star,
                      MPI_COMM_WORLD);

        if (myID == 0) {
            //  Export the global locations list to  .scv files for analyzing in start, middle and end of the
            //  runtime
            FILE *curr_file = NULL;
            if (curr_stage == 0) curr_file = start_file;
            else if (curr_stage == total_stages / 2) curr_file = mid_file;
            else if (curr_stage == total_stages - 1) curr_file = end_file;
            int k;

            if (curr_file) {                                                                   //  Writing to file loop
                for (k = 0; k < N_stars; k++) {
                    fprintf(curr_file, "%lf, %lf\n", global_stars_list[k].location[X],
                            global_stars_list[k].location[Y]);
                }
            }

        }
        //  Update the acceleration, the velocity and the location of each star in the given array
        for (i = 0; i < local_list_length; i++) {
            update_acceleration(&local_stars_list[i], global_stars_list, N_stars);
            update_velocity(&local_stars_list[i], time_step);
            update_location(&local_stars_list[i], time_step);
        }

        curr_stage++;                          //  Increment the stage counter such that the program will run ~2 minutes
    }
    //
    if (myID == 0) {
        endwtime = MPI_Wtime();
        printf("\nExecution time = %f [sec]\n", endwtime - startwtime); //  Print the elapsed execution time
    }

    //  Clean up - free dynamic allocated memory and close files
    free(local_stars_list);
    free(global_stars_list);
    free(local_rand_nums);
    MPI_Type_free(&MPI_Star);                                   //  MPI_Types are only need to be freed once

    if (myID == 0) {

        free(rand_nums);
        fclose(start_file);
        fclose(mid_file);
        fclose(end_file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
/// ------------------------ Function Definitions ------------------------------------------------------------------- //
void initialize_stars(star *stars_list, int length, int ID, const double *random_numbers) {
    ///  Initialize random location and velocity to each star in a given stars list,
    ///  using an input array of random numbers. The random parameters are (X, Y) location and velocity vector
    int i;
    double velocity_vector_size, theta;
    double universe_length = 100 * LIGHT_YEAR;
    for (i = 0; i < length; i++) {
        stars_list[i].acceleration[X] = 0;                                                //  Starting acceleration is 0
        stars_list[i].acceleration[Y] = 0;                                                //  Starting acceleration is 0
        stars_list[i].location[X] = random_numbers[i + 0 * length] * universe_length;
        stars_list[i].location[Y] = random_numbers[i + 1 * length] * universe_length;
        stars_list[i].ID[0] = ID;                                         //  ID[0] is the rank of the creating process,
        stars_list[i].ID[1] = i;                                          //  ID[1] is the index in the local stars list

        velocity_vector_size = random_numbers[i + 2 * length] * AVG_SPEED + 0.5 * AVG_SPEED;
        //  Dividing by RAND_MAX normalizing the random number to [0, 1]
        theta = random_numbers[i + 3 * length];
        stars_list[i].velocity[X] = velocity_vector_size * cos(theta);
        stars_list[i].velocity[Y] = velocity_vector_size * sin(theta);
    }
}

void update_acceleration(struct star *self, const star *other_stars_lists, int star_list_len) {
    ///  Update the acceleration of each star in a given listed by the formula of F = ma,
    ///  where the force is computed out of the gravity forces between the stars
    int i;
    double radius_vector[2], radius_length, coefficient;
    self->acceleration[X] = 0; self->acceleration[Y] = 0; // No accumulation of acceleration from stage to stage

    for (i = 0; i < star_list_len; i++) {              //  Compute the distances between the current point to all others
        radius_vector[X] = other_stars_lists[i].location[X] - self->location[X];
        radius_vector[Y] = other_stars_lists[i].location[Y] - self->location[Y];

        if (!(self->ID[0] == other_stars_lists[i].ID[0] && self->ID[1] == other_stars_lists[i].ID[1])) {
            //  The if condition prevents a star from trying to compute influence on itself
            radius_length = distance(self->location, other_stars_lists[i].location);
            coefficient = (G_FORCE * STAR_MASS / pow(radius_length, 3));

            //  Compute the acceleration out of the equivalent force
            self->acceleration[X] += coefficient * radius_vector[X];
            self->acceleration[Y] += coefficient * radius_vector[Y];
        }
    }
}

void update_velocity(struct star *self, double time_step) {
    ///  Update the velocity of a star in the galaxy, based on the formula v(t) = v_0 + a(t) * t
    self->velocity[X] += self->acceleration[X] * time_step;
    self->velocity[Y] += self->acceleration[Y] * time_step;
}

void update_location(struct star *self, double time_step) {
    ///  Update the location of a star in the galaxy, based on the formula of x(t) = x_0 + v(t) * t
    self->location[X] += self->velocity[X] * time_step;
    self->location[Y] += self->velocity[Y] * time_step;

    //  Apply boundary conditions: if a star is drifted out of the box, let it enter back:
    while (self->location[X] > UNIVERSE_BOUND) self->location[X] -= UNIVERSE_BOUND;
    while (self->location[Y] > UNIVERSE_BOUND) self->location[Y] -= UNIVERSE_BOUND;
    while (self->location[X] < 0) self->location[X] += UNIVERSE_BOUND;
    while (self->location[Y] < 0) self->location[Y] += UNIVERSE_BOUND;
}

double *create_rand_nums(int num_elements) {
    ///  Generate an array of random numbers in range of [0, 1]
    double *rand_nums = (double *) malloc(sizeof(double) * num_elements);
    int i;
    for (i = 0; i < num_elements; i++) {             //  Dividing the random number by RAND_MAX normalizing it to [0, 1]
        rand_nums[i] = rand() / (double) RAND_MAX;   // NOLINT(cert-msc30-c, cert-msc50-cpp)
    }
    return rand_nums;
}

MPI_Datatype create_MPI_star() {
    ///  Define an MPI type MPI_star using the MPI BIFs,
    ///  to enable MPI send and receive MPI_star objects between processes
    int count = 4;
    int blockLens[] = {2, 2, 2, 2};
    MPI_Aint indices[] = {offsetof(star, ID), offsetof(star, location), offsetof(star, acceleration),
                          offsetof(star, velocity)};
    MPI_Datatype old_types[] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Datatype temp_type, new_type;
    MPI_Aint lb, extent;

    MPI_Type_create_struct(count, blockLens, indices,
                           old_types, &temp_type);

    MPI_Type_get_extent(temp_type, &lb, &extent);
    MPI_Type_create_resized(temp_type, lb, extent, &new_type);

    // MPI_Type_create_struct(count, blockLens, indices, old_types, &temp_type);
    MPI_Type_free(&temp_type);
    MPI_Type_commit(&new_type);

    return new_type;
}

double distance(const double *location1, const double *location2) {
    ///  Compute the Euclidean distance between two point in (x, y) plane
    double x1 = location1[0], x2 = location2[0];
    double y1 = location1[1], y2 = location2[1];
    return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
};
