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
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    // This line creates a normal (Gaussian) distribution for x.
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 50;



    for (int i = 0; i < num_particles; ++i) {
        double sample_x, sample_y, sample_theta;
        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);

        Particle temp;
        temp.id = i;
        temp.x = sample_x;
        temp.y = sample_y;
        temp.theta = sample_theta;
        temp.weight = 1;

        weights.push_back(1.0);
        particles.push_back(temp);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    if (fabs(yaw_rate) < 0.0001) {
        yaw_rate = 0.0001;
    }


    for (int i = 0; i < num_particles; i++) {
        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;

        x = x + (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
        y = y + (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
        theta = theta + yaw_rate * delta_t;

        particles[i].x = x + dist_x(gen);
        particles[i].y = y + dist_y(gen);
        particles[i].theta = theta + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    //   observed measurement to this particular landmark.
    //   implement this method and use it as a helper during the updateWeights phase.
    for(int i = 0; i < observations.size(); i++){
        double min_distance = numeric_limits<double>::max();
        for (int j = 0; j < predicted.size(); j++) {
            double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            if(distance < min_distance){
                min_distance = distance;
                observations[i].id = predicted[j].id;
            }
        }

    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    for(int i =0; i < num_particles; i++){
        std::vector<LandmarkObs> transformed_obs;
        Particle current_p = particles[i];

        for(int j = 0; j < observations.size(); j++){

            LandmarkObs obs = observations[j];
            LandmarkObs map_obs;

            map_obs.x = current_p.x + (cos(current_p.theta) * obs.x) - (sin(current_p.theta) * obs.y);
            map_obs.y = current_p.y + (sin(current_p.theta) * obs.x) + (cos(current_p.theta) * obs.y);
            transformed_obs.push_back(map_obs);
        }

        std::vector<LandmarkObs> predicted;
        for(int j = 0; j < map_landmarks.landmark_list.size(); j++){
            float landmark_x = map_landmarks.landmark_list[j].x_f;
            float landmark_y = map_landmarks.landmark_list[j].y_f;
            int landmark_id = map_landmarks.landmark_list[j].id_i;

            double distant = dist(current_p.x, current_p.y, landmark_x, landmark_y);
            if(distant < sensor_range){
                LandmarkObs temp;
                temp.x = landmark_x;
                temp.y = landmark_y;
                temp.id = landmark_id;
                predicted.push_back(temp);
            }
        }

//        for (int j = 0; j < predicted.size(); ++j) {
//            cout<<predicted[j].id<<" ";
//        }
//        cout<<endl;

        if(predicted.size() > 0){
            dataAssociation(predicted, transformed_obs);

            double weight = 1.0;
            for (int j = 0; j < transformed_obs.size(); j++) {
                LandmarkObs predicted_j;
                LandmarkObs observation_j = transformed_obs[j];
                int id = observation_j.id;

                for (int k = 0; k < predicted.size(); ++k) {
                    if(predicted[k].id == id){
                        predicted_j = predicted[k];
                    }
                }
//                cout<<"observation_id"<<observation_j.id<<"predict_id:"<<predicted_j.id<<endl;
//                cout<<observation_j.x<<predicted_j.x<<endl;

                double temp1 = pow(observation_j.x - predicted_j.x, 2) / (2 * pow(std_landmark[0], 2));
                double temp2 = pow(observation_j.y - predicted_j.y, 2) / (2 * pow(std_landmark[1], 2));
                double temp = -(temp1 + temp2);
                weight = weight * (exp(temp) / (2 * M_PI * std_landmark[0], std_landmark[1]));
//                cout<<weight<<endl;
            }
            current_p.weight = weight;
        } else{
            current_p.weight = 0.0;
        }

//        cout<<current_p.weight<<endl;
//        cout<<"predicted size:"<<predicted.size()<<endl;
        weights[i] = current_p.weight;
        particles[i].weight = current_p.weight;



    }
}

void ParticleFilter::resample() {
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    discrete_distribution<> distribution(weights.begin(), weights.end());

    std::vector<Particle> particle_list;
    for (int i = 0; i < num_particles; i++) {
        int number = distribution(gen);
        particle_list.push_back(particles[number]);
    }

//    for (int j = 0; j < num_particles; ++j) {
//        cout<<particle_list[j].id;
//    }
//    cout<<endl;
    particles = particle_list;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
