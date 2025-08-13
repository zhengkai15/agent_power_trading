import src.utils.custom_logger as custom_logger
import os
import chardet # You might need to install this: pip install chardet

def fix_filename_encoding(filename):
    """
    尝试修复文件名编码问题，特别是针对中文乱码。
    """
    # If the filename is already correctly decoded, return it.
    # This is a heuristic: if it contains non-ASCII characters but is valid UTF-8, assume it's correct.
    try:
        filename.encode('utf-8').decode('utf-8')
        return filename
    except UnicodeEncodeError: # This means filename contains characters not representable in UTF-8
        pass # Proceed to try other decodings
    except UnicodeDecodeError: # This should not happen if filename is already a unicode string
        pass

    # Try common encodings for Chinese characters
    encodings_to_try = ['gbk', 'gb2312', 'big5']
    for encoding in encodings_to_try:
        try:
            # Assume the original bytes were in this encoding, and Python decoded it incorrectly.
            # So, re-encode to bytes using latin1 (lossy, but preserves byte values for re-decoding)
            # and then decode with the target encoding.
            decoded_filename = filename.encode('latin1').decode(encoding)
            # Check if the decoded string can be re-encoded to UTF-8 without loss
            if decoded_filename.encode('utf-8').decode('utf-8') == decoded_filename:
                return decoded_filename
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue

    # Fallback to chardet if common encodings fail
    try:
        raw_bytes = filename.encode('latin1') # Still need to get the original bytes if possible
        detection = chardet.detect(raw_bytes)
        encoding = detection['encoding']
        confidence = detection['confidence']
        
        if encoding and confidence > 0.8: # Only use if confidence is high
            decoded_filename = raw_bytes.decode(encoding)
            return decoded_filename
    except Exception:
        pass

    return filename # If all attempts fail, return original filename

def rename_files_in_directory(directory_path):
    """
    遍历指定目录下的所有文件，尝试修复文件名编码并重命名。
    """
    for root, _, files in os.walk(directory_path):
        for filename in files:
            original_path = os.path.join(root, filename)
            fixed_name = fix_filename_encoding(filename)
            if fixed_name != filename:
                new_path = os.path.join(root, fixed_name)
                try:
                    os.rename(original_path, new_path)
                    print(f"Renamed '{original_path}' to '{new_path}'")
                except OSError as e:
                    print(f"Error renaming '{original_path}': {e}")

if __name__ == '__main__':
    # 示例用法：
    # rename_files_in_directory('/path/to/your/data/directory')
    pass
