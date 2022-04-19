#pragma once

#include <core/naive_bayes_trainer.h>
#include <math.h>

namespace naivebayes {

/**
 * A predictor class to classify a image using trained data.
 */
class NaiveBayesPredictor {
 public:

  /**
   * Constructs a Naive Bayes Predictor object given a Naive Bayes Trainer object with trained data and image size.
   *
   * @param trained_model a Naive Bayes Trainer object with trained data
   * @param image_size the size of each image
   */
  NaiveBayesPredictor(NaiveBayesModel& trained_model, size_t image_size);

  /**
   * Constructs a Naive Bayes Predictor object given a input file stream to construct a Naive Bayes Trainer object.
   *
   * @param is_data_trained a boolean value indicating whether the file in the input stream is trained or not
   * @param is the input stream with the file needed
   * @param image_size the size of each image
   * @param training_image_count the total count of images to train
   */
  NaiveBayesPredictor(bool is_data_trained, std::istream& is, size_t image_size, size_t training_image_count);

  /**
   * Reads a image and parse the information into a Naive Bayes Predictor object.
   *
   * @param is the input stream with the image information
   * @param predictor the Naive Bayes Predictor object to construct
   * @return the given input stream
   */
  friend std::istream& operator>>(std::istream& is, NaiveBayesPredictor& predictor);

  std::vector<double>& GetResults();

  /**
   * Classify the given image into a digit with maximum likelihood.
   *
   * @return the predicted digit with maximum likelihood
   */
  size_t Classify();

 private:
  /**
   * The list of likelihood for each candidate digit
   */
  std::vector<double> results_;

  /**
   * The Naive Bayes Trainer object with trained information
   */
  NaiveBayesModel trained_model_;

  /**
   * The list containing the input image information
   */
  std::vector<std::string> image_data_;

  /**
   * The size of the image to classify
   */
  size_t image_size_;

  /**
   * Calculate the likelihood for each candidate digit and construct the results vector.
   */
  void CalculateLikelihood();

  /**
   * Determine whether a coordinate in the image is shaded or not.
   *
   * @param row the row index of the coordinate
   * @param column the column index of the coordinate
   * @return whether the coordinate is shaded
   */
  bool IsShaded(size_t row, size_t column);
};

}