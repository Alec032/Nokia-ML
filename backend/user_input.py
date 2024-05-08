from feature_extraction import process

def process_input(title, description, build, feature, release):
    if not title.strip():
        return "Title cannot be empty."
    
    title_tokens = process(title)

    if not description.strip():
        return "Description cannot be empty."
    
    description_tokens = process(description)

    if not build.strip():
        return "Build cannot be empty."
    
    build_tokens = process(build)

    if not feature.strip():
        return "Feature cannot be empty."
    
    feature_tokens = process(feature)

    if not release.strip():
        return "Release cannot be empty."
    
    release_tokens = process(release)

    combined_feature_tokens = title_tokens + description_tokens
    other_feature_tokens = build_tokens + feature_tokens + release_tokens

    return combined_feature_tokens, other_feature_tokens