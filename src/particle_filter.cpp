/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).


	num_particles = 30;

	default_random_engine gen;
	double std_dev_x = std[0];
	double std_dev_y = std[1];
	double std_dev_theta = std[2];

	normal_distribution<double> dist_x(x, std_dev_x);
	normal_distribution<double> dist_y(y, std_dev_y);
	normal_distribution<double> dist_theta(theta, std_dev_theta);


	for(int i = 0; i < num_particles; i++) {
		Particle particle;

		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);

		particle.weight = 1.0;


		particles.push_back(particle);
		weights.push_back(1.0);

	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	double std_dev_x = std_pos[0];
	double std_dev_y = std_pos[1];
	double std_dev_theta = std_pos[2];

	normal_distribution<double> dist_x(0.0, std_dev_x);
	normal_distribution<double> dist_y(0.0, std_dev_y);
	normal_distribution<double> dist_theta(0.0, std_dev_theta);

	for( auto &particle : particles ) {


		// Car drives straight
		if(fabs(yaw_rate) == 0){
			particle.x += (velocity * cos(particle.theta ) * delta_t);
			particle.y += (velocity * sin(particle.theta ) * delta_t);
		}
		// Car drives in a curve
		else {
			particle.x += (velocity/yaw_rate) * ( sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta) );
			particle.y += (velocity/yaw_rate) * ( cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
			particle.theta += yaw_rate * delta_t;
		}

		// Add noise
		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_theta(gen);

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, Particle &particle) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
//TODO: Check if initialization is required
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;

//	std::vector<int>& associations = getAssociations(particle);
//	std::vector<double>& sense_x = getSenseX(particle);
//	std::vector<double>& sense_y = getSenseY(particle);

	for( unsigned obs_count = 0; obs_count < observations.size(); obs_count++ ) {
//		std::cout << "obs_count : " << obs_count << endl;
		double min_distance = 1.0e99;
		int index, landmark_id;

		landmark_id = -1000;

		double obs_x = observations[obs_count].x;
		double obs_y = observations[obs_count].y;

		// Find the predicted landmark that is closes to the current observation
		for(unsigned pred_count = 0; pred_count < predicted.size(); pred_count++ ) {
			double p_x = predicted[pred_count].x;
			double p_y = predicted[pred_count].y;
			int p_id = predicted[pred_count].id;

			double distance = dist(obs_x, obs_y, p_x, p_y);

			if(distance < min_distance) {
				min_distance = distance;
				index = pred_count;
				landmark_id = p_id;
			}

		}

		if( landmark_id != -1000 ) {
			std::cout << "landmark_id : " << landmark_id << "index : " << index  << endl;

			// Assign the closest landmark to the observation
//			observations[obs_count].id = landmark_id;
			observations[obs_count].id = index;

			// Add the land mark id and the observed x and y values to the particle. This will be used by the simulator to display the observed value associated
			associations.push_back(landmark_id);
			sense_x.push_back(obs_x);
			sense_y.push_back(obs_y);

		}

	}

	std::cout << "Calling SetAssociations" << endl;
	particle = SetAssociations(particle, associations, sense_x, sense_y);
	std::cout << "Associations are set " << endl;

}


double multivariateGaussianProb(double x, double y, double x_mu, double y_mu, double std_landmark[]) {
	double probability;

	double x_val = (( x - x_mu ) * ( x - x_mu )) / (std_landmark[0] * std_landmark[0]);
	double y_val = (( y - y_mu ) * ( y - y_mu )) / (std_landmark[1] * std_landmark[1]);

	probability = exp(-0.5 * ( x_val + y_val ));
	probability /= ( 2 * M_PI * std_landmark[0] * std_landmark[1]);


	return probability;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for(unsigned int particle_count = 0; particle_count < num_particles; particle_count++ ){

		double p_x = particles[particle_count].x;
		double p_y = particles[particle_count].y;
		double p_theta = particles[particle_count].theta;

//		Transform the observations from car co-ordinates to map co-ordinates
		std::vector<LandmarkObs> observations_in_map_coord;
		for(unsigned int obs_count =0; obs_count < observations.size(); obs_count++) {
			double x_map = p_x + cos(p_theta) * observations[obs_count].x - sin(p_theta) * observations[obs_count].y;
			double y_map = p_y + sin(p_theta) * observations[obs_count].x + cos(p_theta) * observations[obs_count].y;

			LandmarkObs current_map_observation = { observations[obs_count].id, x_map, y_map };
			observations_in_map_coord.push_back(current_map_observation);
		}

//		Select Landmarks in sensor range of the current particle
		std::vector<LandmarkObs> landmarks_in_range;
		for(unsigned int landmark_count = 0; landmark_count < map_landmarks.landmark_list.size(); landmark_count++) {
			double lm_x = map_landmarks.landmark_list[landmark_count].x_f;
			double lm_y = map_landmarks.landmark_list[landmark_count].y_f;

			double distance = dist(p_x, p_y, lm_x, lm_y);

			if(distance <= sensor_range) {
				int id = map_landmarks.landmark_list[landmark_count].id_i;
				double x = map_landmarks.landmark_list[landmark_count].x_f;
				double y = map_landmarks.landmark_list[landmark_count].y_f;

				LandmarkObs landmark = {id, x, y};
				landmarks_in_range.push_back(landmark);
			}
		}

//		Associate landmark id to the landmark observations in map co-ordinates
		dataAssociation(landmarks_in_range, observations_in_map_coord, particles[particle_count]);

		std::cout << " landmarks_in_range : " ;
		for(unsigned int count = 0; count < landmarks_in_range.size(); count++) {
			std::cout << landmarks_in_range[count].id << " ";
		}
		std::cout << " #### ";

		std::cout << " observations_in_map_coord : ";
		for(unsigned int count = 0; count < observations_in_map_coord.size(); count++) {
			std::cout << observations_in_map_coord[count].id << " ";
		}
		std::cout << endl;
/*
*/

//		Calculate the particle's final weight
		particles[particle_count].weight = 1.0;
		for(unsigned int count =0; count < observations_in_map_coord.size(); count++) {
			int obs_id = observations_in_map_coord[count].id;
			double obs_x = observations_in_map_coord[count].x;
			double obs_y = observations_in_map_coord[count].y;

			// Find the landmark with the given obs_id
			/*
			for(unsigned int lm_count=0; lm_count <  landmarks_in_range.size(); lm_count++) {
				if(landmarks_in_range[lm_count].id == obs_id) {
					double lm_x = landmarks_in_range[lm_count].x;
					double lm_y = landmarks_in_range[lm_count].y;

					double weight = multivariateGaussianProb(obs_x, obs_y, lm_x, lm_y, std_landmark);
					std::cout << "particles[particle_count].weight : " << particles[particle_count].weight << ", weight : " << weight << endl;
					if( weight > 0.0001) {
						particles[particle_count].weight *= weight;
					}

					break;
				}
			}
			*/

			double lm_x = landmarks_in_range[obs_id].x;
			double lm_y = landmarks_in_range[obs_id].y;

			double weight = multivariateGaussianProb(obs_x, obs_y, lm_x, lm_y, std_landmark);
			std::cout << "particles[particle_count].weight : " << particles[particle_count].weight << ", weight : " << weight << endl;
			if( weight > 0) {
				particles[particle_count].weight *= weight;
			}


		}

		// Update global weights
		weights[particle_count] = particles[particle_count].weight;
	}



}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	std::discrete_distribution<int> d(weights.begin(), weights.end());

	std::vector<Particle> resampled_particles;

	for( unsigned int count = 0; count < num_particles; count++) {
		resampled_particles.push_back(particles[d(gen)]);
	}

	particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int> &associations,
                                     const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

//	particle.associations.clear();
//	particle.sense_x.clear();
//	particle.sense_y.clear();

	std::cout << "Setting particle.associations : " << endl;

    particle.associations= associations;
	std::cout << "particle.associations is set " << endl;
    particle.sense_x = sense_x;
	std::cout << "particle.sense_x is set " << endl;
    particle.sense_y = sense_y;
	std::cout << "particle.sense_y is set " << endl;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
