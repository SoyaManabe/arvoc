from google.cloud import vision
import numpy as np
from explanation import get_explanation_from_gpt
from audio import text_to_speech
import time

file_path = "./labels/IMG_5380.jpg"

def run_quickstart() -> vision.EntityAnnotation:
    """Provides a quick start example for Cloud Vision."""

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # The URI of the image file to annotate
    file_path = "./labels/test.jpeg"

    with open(file_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print("Labels:")
    for label in labels:
        print(label.description)

    return labels


def detect_text(path):

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    texts = response.text_annotations
    #print("Texts:")

    """
    for text in texts:
        print(f'\n"{text.description}"')

        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        ]

        print("bounds: {}".format(",".join(vertices)))
    """

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return response

def get_center(bounding_poly):
    x_coords = [v.x for v in bounding_poly.vertices]
    y_coords = [v.y for v in bounding_poly.vertices]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return np.array([center_x, center_y])

def find_nearest_word(annotate_image_response, target_x, target_y):
    target_point = np.array([target_x, target_y])
    min_distance = float("inf")
    nearest_word = None
    nearest_index = -1
    word_list = []

    text_annotations = annotate_image_response.text_annotations

    for i, annotation in enumerate(text_annotations):
        if annotation.bounding_poly:
            center = get_center(annotation.bounding_poly)
            distance = np.linalg.norm(center - target_point)

            word_list.append(annotation.description)

            if distance < min_distance:
                min_distance = distance
                nearest_word = annotation.description
                nearest_index = i

    return nearest_word, nearest_index, word_list

def extract_sentence_from_nearest(annotate_image_response, target_x, target_y):
    nearest_word, nearest_index, word_list = find_nearest_word(annotate_image_response, target_x, target_y)

    if nearest_word is None:
        return "Sorry, I couldn't find a word at your interest."
    
    sentence = [nearest_word]

    # Backwards
    for i in range(nearest_index + 1, len(word_list)):
        sentence.append(word_list[i])
        if "." in word_list[i]:
            break
    # Forwards
    for i in range(nearest_index -1, -1, -1):
        if "." in word_list[i]:
            break
        sentence.insert(0, word_list[i])
    
    return nearest_word, " ".join(sentence)

def main():
    # time measure start
    start_time = time.time()
    annotate_image_response = detect_text(file_path)
    annotate_time = time.time()
    t1 = annotate_time - start_time
    print(f"finish annotate: {t1:.4f} sec.")
    target_x, target_y = 1300, 1400
    nearest_word, extracted_sentence = extract_sentence_from_nearest(annotate_image_response, target_x, target_y)
    word_search_time = time.time()
    t2 = word_search_time - annotate_time
    print(f"finish word search: {t2:.4f} sec.")
    #print(f"{nearest_word}\n")
    #print(f"{extracted_sentence}\n")
    explanation = get_explanation_from_gpt(nearest_word, extracted_sentence)
    #print(explanation)
    text_to_speech(explanation, lang="ja")
    end_time = time.time()
    t3 = end_time - word_search_time
    print(f"finish explanation: {t3:.4f} sec.")
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.4f} sec.")


if __name__ == "__main__":
    main()