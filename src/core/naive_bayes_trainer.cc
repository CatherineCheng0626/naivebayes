#include <core/naive_bayes_trainer.h>

using std::ostream;
using std::istream;
using std::vector;
using std::string;

namespace naivebayes {

NaiveBayesModel::NaiveBayesModel(size_t image_size, size_t image_count) :
image_size_(image_size), image_count_(image_count) {

  //conditional_probabilities_[row][column][is_shaded][digit 0-9]
  Initialize();
}

void NaiveBayesModel::Initialize() {
  conditional_probabilities_ = vector<vector<vector<vector<double>>>>(image_size_, vector<vector<vector<double>>>(
      image_size_, vector<vector<double>>(2, vector<double>(TYPE_COUNT))));
  prior_indices_ = vector<vector<size_t>>(TYPE_COUNT);
  prior_probabilities_ = vector<double>(TYPE_COUNT);
  image_data_ = vector<vector<string>>(image_count_, vector<string>(image_size_));
}

ostream &operator<<(ostream &os, const NaiveBayesModel &model) {
  os << model.image_size_ << std::endl;
  os << model.image_count_ << std::endl;
  for(size_t i = 0; i < TYPE_COUNT; i++) {
    os << model.prior_probabilities_[i] << std::endl;
  }

  for(size_t k = 0; k < TYPE_COUNT; k++) {
    for(size_t i = 0; i < model.image_size_; i++) {
      for(size_t j = 0; j < model.image_size_; j++){
        os << model.conditional_probabilities_[i][j][1][k] << " ";
      }
      os << std::endl;
    }
    os << std::endl;
  }
  return os;
}

istream &operator>>(istream &is, NaiveBayesModel &model) {
  string line;
  for (size_t i = 0; i < model.image_count_; i++) {
    if (getline(is, line)) {
      model.prior_indices_[stoi(line)].push_back(i);
      for (size_t j = 0; j < model.image_size_; j++) {
        if (getline(is,line)) {
          model.image_data_[i][j] = line;
        }
      }
    }
  }
  model.CalculatePriorProbabilities();
  model.CalculateConditionalProbabilities();
  return is;
}

std::vector<std::vector<std::string>>& NaiveBayesModel::GetImageData() {
  return image_data_;
}

std::vector<std::vector<size_t>>& NaiveBayesModel::GetPriorIndices() {
  return prior_indices_;
}

double NaiveBayesModel::GetPriorProbabilities(size_t prior_digit) {
  return prior_probabilities_[prior_digit];
}

double NaiveBayesModel::GetConditionalProbabilities(size_t row, size_t column, bool is_shaded, size_t prior_digit) {
  return conditional_probabilities_[row][column][is_shaded][prior_digit];
}

void NaiveBayesModel::CalculatePriorProbabilities() {
  for (size_t i = 0; i < TYPE_COUNT; i++) {
    prior_probabilities_[i] = (double) (1 + prior_indices_[i].size()) / (double) (LAPLACE_CONSTANT + image_count_);
  }
}

void NaiveBayesModel::CalculateConditionalProbabilities() {
  for (size_t i = 0; i < image_size_; i++) {
    for (size_t j = 0; j < image_size_; j++) {
      for (size_t k = 0; k < TYPE_COUNT; k++) {
        conditional_probabilities_[i][j][0][k] = CalculateConditionalProbability(i, j,
                                                                                 false, k);
        conditional_probabilities_[i][j][1][k] = CalculateConditionalProbability(i, j,
                                                                                 true, k);
      }
    }
  }
}

double NaiveBayesModel::CalculateConditionalProbability(size_t row, size_t column, bool is_shaded, int class_digit) {
  vector<size_t>& indices = prior_indices_[class_digit];
  size_t count = 0;
  for (size_t index : indices) {
    size_t row_length = image_data_[index][row].size();
    if (is_shaded) {
      if (row_length > column && (image_data_[index][row][column] == '#' || image_data_[index][row][column] == '+')) {
        count++;
      }
    } else {
      if (row_length <= column || image_data_[index][row][column] == ' ') {
        count++;
      }
    }
  }
  return (double) (1 + count) / (double) (2 + prior_indices_[class_digit].size());
}

void NaiveBayesModel::LoadData(std::istream& input_file) {
  string line;
  if(getline(input_file, line)) {
    image_size_ = stoi(line);
  }
  if(getline(input_file, line)) {
    image_count_ = stoi(line);
  }
  Initialize();
  for (size_t i = 0;i < TYPE_COUNT; i++) {
    if (getline(input_file, line)) {
      prior_probabilities_[i] = stod(line);
    }
  }
  for (size_t k = 0; k < TYPE_COUNT; k++) {
    for (size_t i = 0; i < image_size_; i++) {
      if (getline(input_file, line)) {
        std::stringstream ss(line);
        string value;
        for (size_t j = 0; j < image_size_; j++) {
          if (ss >> value) {
            conditional_probabilities_[i][j][1][k] = stod(value);
            conditional_probabilities_[i][j][0][k] = 1 - conditional_probabilities_[i][j][1][k];
          }
        }
      }
    }
    getline(input_file, line);
  }
}

}  // namespace naivebayes