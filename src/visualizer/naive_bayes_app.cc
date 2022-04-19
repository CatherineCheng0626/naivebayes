#include <visualizer/naive_bayes_app.h>

namespace naivebayes {

namespace visualizer {

NaiveBayesApp::NaiveBayesApp()
    : input_file_(kDefaultTrainer),
      predictor_(false, input_file_, kImageDimension, 5000),
      sketchpad_(glm::vec2(kMargin, kMargin), kImageDimension,
                 kWindowSize - 2 * kMargin, predictor_) {
  ci::app::setWindowSize((int) kWindowSize, (int) kWindowSize);
}

void NaiveBayesApp::draw() {
  ci::Color8u background_color(255, 246, 148);  // light yellow
  ci::gl::clear(background_color);

  sketchpad_.Draw();

  ci::gl::drawStringCentered(
      "Press Delete to clear the sketchpad. Press Enter to make a prediction.",
      glm::vec2(kWindowSize / 2, kMargin / 2), ci::Color("black"));

  ci::gl::drawStringCentered(
      "Prediction: " + std::to_string(current_prediction_),
      glm::vec2(kWindowSize / 2, kWindowSize - kMargin / 2), ci::Color("blue"));
}

void NaiveBayesApp::mouseDown(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NaiveBayesApp::mouseDrag(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NaiveBayesApp::keyDown(ci::app::KeyEvent event) {
  if (event.getCode() == KEY_ENTER) {
    std::string image_lines;
    auto& shadings = sketchpad_.GetShadings();

    for (size_t row = 0; row < kImageDimension; ++row) {
      for (size_t col = 0; col < kImageDimension; ++col) {
        if (!shadings[row][col]) {
          image_lines += " ";
        } else {
          image_lines += SHADED_CHAR;
        }
      }
      image_lines += "\n";
    }
    std::istringstream inputs(image_lines);
    inputs >> predictor_;
    current_prediction_ = predictor_.Classify();
  }

  if (event.getCode() == KEY_DELETE) {
    sketchpad_.Clear();
  }
}

}  // namespace visualizer

}  // namespace naivebayes
