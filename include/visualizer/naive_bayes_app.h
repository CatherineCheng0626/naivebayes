#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "sketchpad.h"
#include <core/naive_bayes_predictor.h>

#define KEY_DELETE 8
#define KEY_ENTER 13
#define SHADED_CHAR "#"

namespace naivebayes {

namespace visualizer {

/**
 * Allows a user to draw a digit on a sketchpad and uses Naive Bayes to
 * classify it.
 */
class NaiveBayesApp : public ci::app::App {
 public:
  NaiveBayesApp();

  void draw() override;
  void mouseDown(ci::app::MouseEvent event) override;
  void mouseDrag(ci::app::MouseEvent event) override;
  void keyDown(ci::app::KeyEvent event) override;

  // provided that you can see the entire UI on your screen.
  const double kWindowSize = 800;
  const double kMargin = 75;
  const size_t kImageDimension = 28;


 private:
  /**
   * The file path of the training data
   */
  const char *kDefaultTrainer = "../../../../../../mnistdatatraining/sample_5000_images.txt";

  /**
   * The input stream of the training data
   */
  std::ifstream input_file_;

  /**
   * The Naive Bayes Predictor object to predict the digit
   */
  NaiveBayesPredictor predictor_;

  /**
   * The sketchpad for visualization
   */
  Sketchpad sketchpad_;

  /**
   * The current digit prediction based on the sketch
   */
  int current_prediction_ = -1;
};

}  // namespace visualizer

}  // namespace naivebayes
