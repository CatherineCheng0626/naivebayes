#include <catch2/catch.hpp>
#include <vector>
#include <iostream>

#include <core/naive_bayes_trainer.h>
#include <core/naive_bayes_predictor.h>

using namespace naivebayes;
using std::string;

TEST_CASE("Test Classifying Given Digit") {
  std::string root_directory = "/Users/catherinecheng/cinder_0.9.2_mac/my-projects/naivebayes-CatherineCheng0626/";

  SECTION("Successfully classifying a image given a training file of one image") {
    std::string input_file_name = root_directory + "mnistdatatraining/sample_one_image.txt";
    std::ifstream input_file(input_file_name);
    NaiveBayesModel naive_bayes_model(5, 1);
    input_file >> naive_bayes_model;
    REQUIRE(input_file.is_open());

    NaiveBayesPredictor naive_bayes_predictor(naive_bayes_model, 5);
    std::string predict_file_name = root_directory + "data_evaluating/test_one_image.txt";
    std::ifstream predict_file(predict_file_name);
    REQUIRE(predict_file.is_open());

    for (size_t count = 0; count < 1; count++) {
      string input_lines;
      string line;

      getline(predict_file, line);

      for (size_t i = 0; i < 5; i++) {
        getline(predict_file, line);
        input_lines += line + "\n";
      }
      std::istringstream input_stream(input_lines);
      input_stream >> naive_bayes_predictor;
      naive_bayes_predictor.Classify();
      auto& results = naive_bayes_predictor.GetResults();

      REQUIRE(-8.5671425 == Approx(results[0]));
      REQUIRE(-6.3467641 == Approx(results[1]));
      REQUIRE(-8.5671425 == Approx(results[2]));
      REQUIRE(-8.5671425 == Approx(results[3]));
      REQUIRE(-8.5671425 == Approx(results[4]));
      REQUIRE(-8.5671425 == Approx(results[5]));
      REQUIRE(-8.5671425 == Approx(results[6]));
      REQUIRE(-8.5671425 == Approx(results[7]));
      REQUIRE(-8.5671425 == Approx(results[8]));
      REQUIRE(-8.5671425 == Approx(results[9]));
    }

    predict_file.close();
    input_file.close();
  }

  SECTION("Successfully classifying a image given a training file of more images") {
    std::string input_file_name = root_directory + "mnistdatatraining/sample_more_images.txt";
    std::ifstream input_file(input_file_name);
    NaiveBayesModel naive_bayes_model(5, 3);
    input_file >> naive_bayes_model;
    REQUIRE(input_file.is_open());

    NaiveBayesPredictor naive_bayes_predictor(naive_bayes_model, 5);
    std::string predict_file_name = root_directory + "data_evaluating/test_one_image.txt";
    std::ifstream predict_file(predict_file_name);
    REQUIRE(predict_file.is_open());

    for (size_t count = 0; count < 1; count++) {
      string input_lines;
      string line;

      getline(predict_file, line);

      for (size_t i = 0; i < 5; i++) {
        getline(predict_file, line);
        input_lines += line + "\n";
      }
      std::istringstream input_stream(input_lines);
      input_stream >> naive_bayes_predictor;
      naive_bayes_predictor.Classify();
      auto& results = naive_bayes_predictor.GetResults();

      REQUIRE(-7.9244648 == Approx(results[0]));
      REQUIRE(-5.8448668 == Approx(results[1]));
      REQUIRE(-8.6396932 == Approx(results[2]));
      REQUIRE(-8.6396932 == Approx(results[3]));
      REQUIRE(-8.6396932 == Approx(results[4]));
      REQUIRE(-8.6396932 == Approx(results[5]));
      REQUIRE(-8.6396932 == Approx(results[6]));
      REQUIRE(-8.6396932 == Approx(results[7]));
      REQUIRE(-8.6396932 == Approx(results[8]));
      REQUIRE(-8.6396932 == Approx(results[9]));
    }

    predict_file.close();
    input_file.close();
  }
}