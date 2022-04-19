#include <catch2/catch.hpp>
#include <vector>
#include <iostream>

#include <core/naive_bayes_trainer.h>
using namespace naivebayes;
using std::vector;

TEST_CASE("Test Reading file") {
  std::string root_directory = "/Users/catherinecheng/cinder_0.9.2_mac/my-projects/naivebayes-CatherineCheng0626/";

  SECTION("Successfully reading one image") {
    std::string input_file_name = root_directory + "mnistdatatraining/sample_one_image.txt";
    std::ifstream input_file(input_file_name);
    REQUIRE(input_file.is_open());

    NaiveBayesModel naive_bayes_model(5, 1);
    input_file >> naive_bayes_model;
    REQUIRE(1 == naive_bayes_model.GetImageData().size());
    REQUIRE(5 == naive_bayes_model.GetImageData()[0].size());
    REQUIRE(10 == naive_bayes_model.GetPriorIndices().size());
    REQUIRE(1 == naive_bayes_model.GetPriorIndices()[1].size());

    input_file.close();
  }

  SECTION("Successfully reading more images") {
    std::string input_file_name = root_directory + "mnistdatatraining/sample_more_images.txt";
    std::ifstream input_file(input_file_name);
    REQUIRE(input_file.is_open());

    NaiveBayesModel naive_bayes_model(5, 3);
    input_file >> naive_bayes_model;
    REQUIRE(3 == naive_bayes_model.GetImageData().size());
    REQUIRE(5 == naive_bayes_model.GetImageData()[0].size());
    REQUIRE(10 == naive_bayes_model.GetPriorIndices().size());
    REQUIRE(1 == naive_bayes_model.GetPriorIndices()[0].size());
    REQUIRE(2 == naive_bayes_model.GetPriorIndices()[1].size());

    input_file.close();
  }
}

TEST_CASE("Test Calculating Prior Class Probability") {
  std::string root_directory = "/Users/catherinecheng/cinder_0.9.2_mac/my-projects/naivebayes-CatherineCheng0626/";

  SECTION("Successfully calculating prior class probability when adding one image") {
    std::string input_file_name = root_directory + "mnistdatatraining/sample_one_image.txt";
    std::ifstream input_file(input_file_name);
    REQUIRE(input_file.is_open());

    NaiveBayesModel naive_bayes_model(5, 1);
    input_file >> naive_bayes_model;

    REQUIRE(1.0/11.0 == naive_bayes_model.GetPriorProbabilities(0));
    REQUIRE(2.0/11.0 == naive_bayes_model.GetPriorProbabilities(1));
    REQUIRE(1.0/11.0 == naive_bayes_model.GetPriorProbabilities(2));
    REQUIRE(1.0/11.0 == naive_bayes_model.GetPriorProbabilities(3));
    REQUIRE(1.0/11.0 == naive_bayes_model.GetPriorProbabilities(4));
    REQUIRE(1.0/11.0 == naive_bayes_model.GetPriorProbabilities(5));
    REQUIRE(1.0/11.0 == naive_bayes_model.GetPriorProbabilities(6));
    REQUIRE(1.0/11.0 == naive_bayes_model.GetPriorProbabilities(7));
    REQUIRE(1.0/11.0 == naive_bayes_model.GetPriorProbabilities(8));
    REQUIRE(1.0/11.0 == naive_bayes_model.GetPriorProbabilities(9));
    input_file.close();
  }

  SECTION("Successfully calculating prior class probability when adding more images") {
    std::string input_file_name = root_directory + "mnistdatatraining/sample_more_images.txt";
    std::ifstream input_file(input_file_name);
    REQUIRE(input_file.is_open());

    NaiveBayesModel naive_bayes_model(5, 3);
    input_file >> naive_bayes_model;

    REQUIRE(2.0/13.0 == naive_bayes_model.GetPriorProbabilities(0));
    REQUIRE(3.0/13.0 == naive_bayes_model.GetPriorProbabilities(1));
    REQUIRE(1.0/13.0 == naive_bayes_model.GetPriorProbabilities(2));
    REQUIRE(1.0/13.0 == naive_bayes_model.GetPriorProbabilities(3));
    REQUIRE(1.0/13.0 == naive_bayes_model.GetPriorProbabilities(4));
    REQUIRE(1.0/13.0 == naive_bayes_model.GetPriorProbabilities(5));
    REQUIRE(1.0/13.0 == naive_bayes_model.GetPriorProbabilities(6));
    REQUIRE(1.0/13.0 == naive_bayes_model.GetPriorProbabilities(7));
    REQUIRE(1.0/13.0 == naive_bayes_model.GetPriorProbabilities(8));
    REQUIRE(1.0/13.0 == naive_bayes_model.GetPriorProbabilities(9));
    input_file.close();
  }
}

TEST_CASE("Test Calculating Conditional Probability") {
  std::string root_directory = "/Users/catherinecheng/cinder_0.9.2_mac/my-projects/naivebayes-CatherineCheng0626/";

  SECTION("Successfully calculating conditional probability when adding one image") {
    std::string input_file_name = root_directory + "mnistdatatraining/sample_one_image.txt";
    std::ifstream input_file(input_file_name);
    REQUIRE(input_file.is_open());

    NaiveBayesModel naive_bayes_model(5, 1);
    input_file >> naive_bayes_model;

    vector<vector<vector<vector<double>>>> result(5, vector<vector<vector<double>>>(
        5, vector<vector<double>>(2, vector<double>(TYPE_COUNT))));

    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        for (size_t k = 0; k < TYPE_COUNT; k++) {
          result[i][j][0][k] = 1.0 / 2.0;
          result[i][j][1][k] = 1.0 / 2.0;
        }
      }
    }
    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        result[i][j][0][1] = 2.0 / 3.0;
        result[i][j][1][1] = 1.0 / 3.0;
      }
    }
    result[0][2][0][1] = 1.0 / 3.0;
    result[1][2][0][1] = 1.0 / 3.0;
    result[2][2][0][1] = 1.0 / 3.0;
    result[3][2][0][1] = 1.0 / 3.0;
    result[0][2][1][1] = 2.0 / 3.0;
    result[1][2][1][1] = 2.0 / 3.0;
    result[2][2][1][1] = 2.0 / 3.0;
    result[3][2][1][1] = 2.0 / 3.0;

    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        for (size_t k = 0; k < TYPE_COUNT; k++) {
          REQUIRE(result[i][j][0][k] == naive_bayes_model.GetConditionalProbabilities(i, j,
                                                                                      false, k));
          REQUIRE(result[i][j][1][k] == naive_bayes_model.GetConditionalProbabilities(i, j,
                                                                                      true, k));
        }
      }
    }
  }

  SECTION("Successfully calculating conditional probability when adding more images") {
    std::string input_file_name = root_directory + "mnistdatatraining/sample_more_images.txt";
    std::ifstream input_file(input_file_name);
    REQUIRE(input_file.is_open());

    NaiveBayesModel naive_bayes_model(5, 3);
    input_file >> naive_bayes_model;

    vector<vector<vector<vector<double>>>> result(5, vector<vector<vector<double>>>(
        5, vector<vector<double>>(2, vector<double>(TYPE_COUNT))));

    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        for (size_t k = 2; k < TYPE_COUNT; k++) {
          result[i][j][0][k] = 1.0 / 2.0;
          result[i][j][1][k] = 1.0 / 2.0;
        }
      }
    }
    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        result[i][j][0][0] = 2.0 / 3.0;
        result[i][j][1][0] = 1.0 / 3.0;
      }
    }
    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        result[i][j][0][1] = 3.0 / 4.0;
        result[i][j][1][1] = 1.0 / 4.0;
      }
    }

    result[0][1][0][0] = 1.0 / 3.0;
    result[1][1][0][0] = 1.0 / 3.0;
    result[2][1][0][0] = 1.0 / 3.0;
    result[0][1][1][0] = 2.0 / 3.0;
    result[1][1][1][0] = 2.0 / 3.0;
    result[2][1][1][0] = 2.0 / 3.0;

    result[0][2][0][1] = 1.0 / 4.0;
    result[1][2][0][1] = 1.0 / 4.0;
    result[2][2][0][1] = 1.0 / 4.0;
    result[3][2][0][1] = 2.0 / 4.0;
    result[0][2][1][1] = 3.0 / 4.0;
    result[1][2][1][1] = 3.0 / 4.0;
    result[2][2][1][1] = 3.0 / 4.0;
    result[3][2][1][1] = 2.0 / 4.0;

    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        for (size_t k = 0; k < TYPE_COUNT; k++) {
          REQUIRE(result[i][j][0][k] == naive_bayes_model.GetConditionalProbabilities(i, j,
                                                                                      false, k));
          REQUIRE(result[i][j][1][k] == naive_bayes_model.GetConditionalProbabilities(i, j,
                                                                                      true, k));
        }
      }
    }
  }
}

TEST_CASE("Test Saving Trained Data") {
  std::string root_directory = "/Users/catherinecheng/cinder_0.9.2_mac/my-projects/naivebayes-CatherineCheng0626/";
  SECTION("Successfully save one-image trained data") {
    std::string output_file_name = root_directory + "mnistdatatraining/one_image_actual_output.txt";
    std::string input_file_name = root_directory + "mnistdatatraining/sample_one_image.txt";
    std::string expected_file_name = root_directory + "mnistdatatraining/expected_output_one.txt";

    std::ifstream input_file(input_file_name);
    std::ofstream output_file(output_file_name);
    REQUIRE(input_file.is_open());
    REQUIRE(output_file.is_open());

    NaiveBayesModel model_to_compare(5, 1);
    input_file >> model_to_compare;
    output_file << model_to_compare;

    std::ifstream actual(output_file_name);
    std::ifstream expected(expected_file_name);
    REQUIRE(actual.is_open());
    REQUIRE(expected.is_open());

    std::string line_actual;
    std::string line_expected;
    while (getline(actual, line_actual)) {
      getline(expected, line_expected);
      REQUIRE(line_expected == line_actual);
    }
  }

  SECTION("Successfully save more-images trained data") {
    std::string output_file_name = root_directory + "mnistdatatraining/more_images_actual_output.txt";
    std::string input_file_name = root_directory + "mnistdatatraining/sample_more_images.txt";
    std::string expected_file_name = root_directory + "mnistdatatraining/expected_output_more.txt";

    std::ifstream input_file(input_file_name);
    std::ofstream output_file(output_file_name);
    REQUIRE(input_file.is_open());
    REQUIRE(output_file.is_open());

    NaiveBayesModel model_to_compare(5, 3);
    input_file >> model_to_compare;
    output_file << model_to_compare;

    std::ifstream actual(output_file_name);
    std::ifstream expected(expected_file_name);
    REQUIRE(actual.is_open());
    REQUIRE(expected.is_open());

    std::string line_actual;
    std::string line_expected;
    while (getline(actual, line_actual)) {
      getline(expected, line_expected);
      REQUIRE(line_expected == line_actual);
    }
  }
}

TEST_CASE("Test Loading Trained Data") {
  std::string root_directory = "/Users/catherinecheng/cinder_0.9.2_mac/my-projects/naivebayes-CatherineCheng0626/";
  SECTION("Successfully load one-image trained data") {
    std::string loading_file_name = root_directory + "mnistdatatraining/output_one.txt";
    std::string input_file_name = root_directory + "mnistdatatraining/sample_one_image.txt";
    std::ifstream input_file(input_file_name);
    std::ifstream load_file(loading_file_name);
    REQUIRE(input_file.is_open());
    REQUIRE(load_file.is_open());

    NaiveBayesModel model_to_compare(5, 1);
    input_file >> model_to_compare;

    NaiveBayesModel naive_bayes_loaded(5, 1);
    naive_bayes_loaded.LoadData(load_file);

    for (size_t i = 0; i < TYPE_COUNT; i++) {
      REQUIRE(model_to_compare.GetPriorProbabilities(i) ==
      Approx(naive_bayes_loaded.GetPriorProbabilities(i)));
    }

    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        for (size_t k = 0; k < TYPE_COUNT; k++) {
          REQUIRE(model_to_compare.GetConditionalProbabilities(i, j,
                                                               false, k) ==
          Approx(naive_bayes_loaded.GetConditionalProbabilities(i, j,
                                                                false, k)));

          REQUIRE(model_to_compare.GetConditionalProbabilities(i, j,
                                                               true, k) ==
              Approx(naive_bayes_loaded.GetConditionalProbabilities(i, j,
                                                                    true, k)));
        }
      }
    }
  }

  SECTION("Successfully load more-images trained data") {
    std::string loading_file_name = root_directory + "mnistdatatraining/output_more.txt";
    std::string input_file_name = root_directory + "mnistdatatraining/sample_more_images.txt";
    std::ifstream input_file(input_file_name);
    std::ifstream load_file(loading_file_name);
    REQUIRE(input_file.is_open());
    REQUIRE(load_file.is_open());

    NaiveBayesModel model_to_compare(5, 3);
    input_file >> model_to_compare;

    NaiveBayesModel naive_bayes_loaded(5, 3);
    naive_bayes_loaded.LoadData(load_file);

    for (size_t i = 0; i < TYPE_COUNT; i++) {
      REQUIRE(model_to_compare.GetPriorProbabilities(i) ==
      Approx(naive_bayes_loaded.GetPriorProbabilities(i)));
    }

    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        for (size_t k = 0; k < TYPE_COUNT; k++) {
          REQUIRE(model_to_compare.GetConditionalProbabilities(i, j,
                                                               false, k) ==
              Approx(naive_bayes_loaded.GetConditionalProbabilities(i, j,
                                                                    false, k)));

          REQUIRE(model_to_compare.GetConditionalProbabilities(i, j,
                                                               true, k) ==
              Approx(naive_bayes_loaded.GetConditionalProbabilities(i, j,
                                                                    true, k)));
        }
      }
    }
  }
}


