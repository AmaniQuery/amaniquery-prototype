import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
} from 'react-native';
import {
  pick,
  isErrorWithCode,
  errorCodes,
  types,
  type DocumentPickerResponse,
} from '@react-native-documents/picker';
import Icon from 'react-native-vector-icons/Ionicons';

interface FilePickerProps {
  files: DocumentPickerResponse[];
  onFilesChange: (files: DocumentPickerResponse[]) => void;
  maxFiles?: number;
  maxSizeMB?: number;
}

export const FilePicker: React.FC<FilePickerProps> = ({
  files,
  onFilesChange,
  maxFiles = 5,
  maxSizeMB = 10,
}) => {
  const pickDocument = async () => {
    try {
      const results = await pick({
        type: [types.pdf, types.images, types.plainText],
        allowMultiSelection: true,
      });

      const maxSize = maxSizeMB * 1024 * 1024;
      const newFiles: DocumentPickerResponse[] = [];

      for (const file of results) {
        // Check file count
        if (files.length + newFiles.length >= maxFiles) {
          Alert.alert('File Limit', `Maximum ${maxFiles} files allowed`);
          break;
        }

        // Check file size
        if (file.size && file.size > maxSize) {
          Alert.alert(
            'File Too Large',
            `${file.name} exceeds ${maxSizeMB}MB limit`,
          );
          continue;
        }

        // Check file type
        const ext = file.name?.split('.').pop()?.toLowerCase();
        const allowedExts = ['pdf', 'png', 'jpg', 'jpeg', 'txt', 'md'];
        if (!ext || !allowedExts.includes(ext)) {
          Alert.alert(
            'Invalid File Type',
            `File type not supported: ${file.name}`,
          );
          continue;
        }

        newFiles.push(file);
      }

      if (newFiles.length > 0) {
        onFilesChange([...files, ...newFiles]);
      }
    } catch (err) {
      if (isErrorWithCode(err) && err.code === errorCodes.OPERATION_CANCELED) {
        // User cancelled
      } else {
        Alert.alert('Error', 'Failed to pick document');
      }
    }
  };

  const removeFile = (index: number) => {
    const newFiles = files.filter((_, i) => i !== index);
    onFilesChange(newFiles);
  };

  const getFileIcon = (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase();
    if (ext === 'pdf') {
      return 'document-text';
    } else if (['png', 'jpg', 'jpeg'].includes(ext || '')) {
      return 'image';
    } else {
      return 'document';
    }
  };

  const formatFileSize = (bytes?: number | null) => {
    if (!bytes) return 'Unknown size';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.pickerButton}
        onPress={pickDocument}
        disabled={files.length >= maxFiles}>
        <Icon name="attach" size={20} color="#007AFF" />
        <Text style={styles.pickerButtonText}>
          {files.length >= maxFiles
            ? `Maximum ${maxFiles} files`
            : 'Attach File'}
        </Text>
      </TouchableOpacity>

      {files.length > 0 && (
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.filesContainer}>
          {files.map((file, index) => (
            <View key={index} style={styles.fileItem}>
              <Icon
                name={getFileIcon(file.name || '')}
                size={24}
                color="#007AFF"
              />
              <View style={styles.fileInfo}>
                <Text style={styles.fileName} numberOfLines={1}>
                  {file.name}
                </Text>
                <Text style={styles.fileSize}>{formatFileSize(file.size)}</Text>
              </View>
              <TouchableOpacity
                onPress={() => removeFile(index)}
                style={styles.removeButton}>
                <Icon name="close-circle" size={20} color="#FF3B30" />
              </TouchableOpacity>
            </View>
          ))}
        </ScrollView>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 8,
  },
  pickerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#F0F0F0',
    borderRadius: 8,
    marginBottom: 8,
  },
  pickerButtonText: {
    marginLeft: 8,
    color: '#007AFF',
    fontSize: 14,
    fontWeight: '500',
  },
  filesContainer: {
    marginTop: 4,
  },
  fileItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F5F5F5',
    borderRadius: 8,
    padding: 8,
    marginRight: 8,
    minWidth: 150,
    maxWidth: 200,
  },
  fileInfo: {
    flex: 1,
    marginLeft: 8,
    marginRight: 8,
  },
  fileName: {
    fontSize: 12,
    fontWeight: '500',
    color: '#000',
  },
  fileSize: {
    fontSize: 10,
    color: '#666',
    marginTop: 2,
  },
  removeButton: {
    padding: 4,
  },
});
