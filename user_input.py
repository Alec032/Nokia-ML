from feature_extraction import process

def process_input():
    title = input("Enter title: ")
    while not title:
        print("Title cannot be empty.")
        title = input("Enter title: ")
    
    title_tokens = process(title)

    description = input("Enter description: ")
    while not description:
        print("Description cannot be empty.")
        description = input("Enter description: ")
    
    description_tokens = process(description)

    build = input("Enter build: ")
    while not build:
        print("Build cannot be empty.")
        build = input("Enter build: ")
    
    build_tokens = process(build)

    feature = input("Enter feature: ")
    while not feature:
        print("Feature cannot be empty.")
        feature = input("Enter feature: ")
    
    feature_tokens = process(feature)

    release = input("Enter release: ")
    while not release:
        print("Release cannot be empty.")
        release = input("Enter release: ")
    
    release_tokens = process(release)

    combined_feature_tokens = title_tokens + description_tokens
    other_feature_tokens = build_tokens + feature_tokens + release_tokens

    with open("features.txt", "a") as features_file:
        features_file.write(" ".join(combined_feature_tokens) + "\n")

    with open("other_features.txt", "a") as other_features_file:
        other_features_file.write(" ".join(other_feature_tokens) + "\n")
