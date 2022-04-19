#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#define TYPE_COUNT 10
#define LAPLACE_CONSTANT 10

namespace naivebayes {

/**
 * The class using Naive Bayes to train the data to recognize the number in a given picture.
 */
class NaiveBayesModel {

 public:

  /**
   * Construct a Naive Bayes Trainer object given the image size and number of images in the untrained data,
   *
   * @param image_size the size of each image
   * @param image_count the total count of images in the data.
   */
  NaiveBayesModel(size_t image_size, size_t image_count);

  /**
   * Outputs the information in the Naive Bayes Trainer into a file.
   *
   * @param os the output stream containing the information for the output file.
   * @param model the Naive Bayes Trainer object to output from.
   * @return the output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const NaiveBayesModel& model);

  /**
   * Reads information from an untrained data to construct the Naive Bayes Trainer object.
   *
   * @param os the input stream containing an untrained data file.
   * @param model the Naive Bayes Trainer object to construct.
   * @return the input stream.
   */
  friend std::istream& operator>>(std::istream& is, NaiveBayesModel& model);

  std::vector<std::vector<std::string>>& GetImageData();

  std::vector<std::vector<size_t>>& GetPriorIndices();

  double GetPriorProbabilities(size_t prior_digit);

  double GetConditionalProbabilities(size_t row, size_t column, bool is_shaded, size_t prior_digit);

  /**
   * Loads the information of a Naive Bayes Trainer given a trained output file.
   *
   * @param input_file the input stream containing the input file.
   */
  void LoadData(std::istream& input_file);

 private:
  /**
   * The size of the image.
   */
  size_t image_size_;

  /**
   * The overall count of the images in the Naive Bayes Trainer.
   */
  size_t image_count_;

  /**
   * The 2-D vector recording the image data strings.
   */
  std::vector<std::vector<std::string>> image_data_;

  /**
   * The list that maps each prior class digit to the indices in the image data that contains the data for that digit.
   */
  std::vector<std::vector<size_t>> prior_indices_;

  /**
   * The vector recording the calculated prior class probabilities for each digit.
   */
  std::vector<double> prior_probabilities_;

  /**
   * The vector recording the conditional probability for each grid (row, column), whether it is shaded, and the
   * prior class digit.
   * NOTE: conditional_probabilities_[row][column][is_shaded][prior_class_digit]
   */
  std::vector<std::vector<std::vector<std::vector<double>>>> conditional_probabilities_;

  /**
   * Initializes the sizes for the class's multidimensional vector variables.
   */
  void Initialize();

  /**
   * Calculate the prior class probabilities for the Naive Bayes Trainer.
   */
  void CalculatePriorProbabilities();

  /**
   * Calculate the conditional probabilities for the Naive Bayes Trainer.
   */
  void CalculateConditionalProbabilities();

  /**
   * Calculate one conditional probability given the row number, column number, whether it is shaded or not,
   * and the class digit to calculate.
   *
   * @param row the row number of the target.
   * @param column the column number of the target.
   * @param is_shaded the boolean variable indicating whether the target is shaded.
   * @param class_digit the prior class digit to calculate.
   * @return
   */
  double CalculateConditionalProbability(size_t row, size_t column, bool is_shaded, int class_digit);
};

}  // namespace naivebayes
