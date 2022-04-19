#include <iostream>

#include <core/naive_bayes_trainer.h>
#include <core/naive_bayes_predictor.h>


using namespace naivebayes;
using std::string;

int main(int argc, char* argv[]) {
  if (argc != 5) {
    return 0;
  }

  std::string input_file_name = std::string(argv[3]);
  std::string predict_file_name = std::string(argv[4]);

  NaiveBayesModel naive_bayes_model(atoi(argv[1]), atoi(argv[2]));
  std::ifstream input_file(input_file_name);

  if (input_file.is_open()) {
    input_file >> naive_bayes_model;
  } else {
    std::cerr << "Fail reading training file" << std::endl;
    return 0;
  }

  NaiveBayesPredictor naive_bayes_predictor(naive_bayes_model, 28);
  std::ifstream predict_file(predict_file_name);

  if (!predict_file.is_open()) {
    std::cerr << "Fail reading testing file" << std::endl;
    return 0;
  }

  size_t correct = 0;

  for (size_t count = 0; count < 1000; count++) {
    string lines;
    string line;

    getline(predict_file, line);
    size_t expected_digit = stoi(line);

    for (size_t i = 0; i < 28; i++) {
      getline(predict_file, line);
      lines += line + "\n";
    }
    std::istringstream is(lines);
    is >> naive_bayes_predictor;
    size_t ind = naive_bayes_predictor.Classify();
    if (ind == expected_digit) {
      correct++;
    }
  }
  double accuracy = correct / 1000.0;
  std::cout << "Accuracy: " << accuracy << std::endl;

  input_file.close();
  predict_file.close();
  return 0;
}
