#include <core/naive_bayes_predictor.h>

using std::string;
using std::vector;

namespace naivebayes {

NaiveBayesPredictor::NaiveBayesPredictor(NaiveBayesModel& trained_model, size_t image_size) :
    trained_model_(trained_model), image_size_(image_size) {
  results_.resize(TYPE_COUNT);
  image_data_.resize(image_size_);
}

NaiveBayesPredictor::NaiveBayesPredictor(bool is_data_trained, std::istream &is, size_t image_size,
                                         size_t training_image_count) :
                                         trained_model_(image_size, training_image_count),
                                         image_size_(image_size) {
  if (is_data_trained) {
    trained_model_.LoadData(is);
  } else {
    is >> trained_model_;
  }
  results_.resize(TYPE_COUNT);
  image_data_.resize(image_size_);
}

std::istream &operator>>(std::istream &is, NaiveBayesPredictor &predictor) {
  string line;
  predictor.image_data_.clear();
  predictor.image_data_.resize(predictor.image_size_);
  for (size_t i = 0; i < predictor.image_size_; i++) {
    if (getline(is, line)) {
      predictor.image_data_[i] = line;
    }
  }
  return is;
}

std::vector<double>& NaiveBayesPredictor::GetResults() {
  return results_;
}

bool NaiveBayesPredictor::IsShaded(size_t row, size_t column) {
  size_t row_length = image_data_[row].size();
  return row_length > column && (image_data_[row][column] == '#' || image_data_[row][column] == '+');
}

void NaiveBayesPredictor::CalculateLikelihood() {
  for (size_t prior_digit = 0; prior_digit < TYPE_COUNT; prior_digit++) {
    double likelihood = log10(trained_model_.GetPriorProbabilities(prior_digit));
    for (size_t row = 0; row < image_size_; row++) {
      for (size_t column = 0; column < image_size_; column++) {
        likelihood += log10(trained_model_.GetConditionalProbabilities(row, column, IsShaded(row, column), prior_digit));
      }
    }

    results_[prior_digit] = likelihood;
  }
}

size_t NaiveBayesPredictor::Classify() {
  CalculateLikelihood();
  double max_likelihood = INT_MIN;
  size_t max_index = -1;
  for (size_t i = 0; i < TYPE_COUNT; i++) {
    if (results_[i] > max_likelihood) {
      max_likelihood = results_[i];
      max_index = i;
    }
  }
  return max_index;
}

}