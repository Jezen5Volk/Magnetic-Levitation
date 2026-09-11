import numpy as np
import cv2
import os



class Videos:

    def __init__(self, video_directory): 
        self.video_directory = video_directory



    def analyze_directory(self, analysis_function): 
        '''
        Walk through the directory, applying the analysis function to every video
        in the directory
        '''

        for root, _, files in os.walk(self.video_directory):
            for file in files: 
                analysis_function(root + file)

        return 


    def compute_rpm(self, video_path, fps): 
        '''
        Analyze video footage to identify the rotations per minute of a rotating object
        with a mark applied to track rotational movement
        '''


        capture = cv2.VideoCapture(video_path)

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        rpm = []
        frames_1sec = np.empty((height, width, fps))
        counter = 0
        while True: 
            ret, frame = capture.read()

            #end condition
            if not ret: 
                break

            #convert to grayscale
            frames_1sec[:, :, counter] = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) 

            if (counter+1)%fps == 0: 
                counter = 0

                #compute threshold
                var = np.var(frames_1sec, axis = -1)
                var_norm = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, thresh = cv2.threshold(var_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

                #Find centroid with max variance peak
                y_peak, x_peak = np.unravel_index(np.argmax(var), var.shape)
                peak_label = labels[y_peak, x_peak]
                centroid_mask = (labels == peak_label)
                centroid_y, centroid_x = np.where(centroid_mask == True)

                #get outline of centroid (for visualization) 
                centroid_mask_uint8 = (centroid_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(centroid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    marker_outline = max(contours, key=cv2.contourArea)

                #compute ffts
                zero_pad_multiplier = 4
                hanning = np.hanning(fps)
                windowed_fft = (frames_1sec[centroid_y, centroid_x, :] - np.mean(frames_1sec[centroid_y, centroid_x, :]))*hanning
                fft = np.mean(np.fft.rfft(windowed_fft, axis = -1, n = fps*zero_pad_multiplier), axis = 0)
                log_fft = np.log(np.abs(fft))
                cepstrum = np.fft.irfft(log_fft - np.mean(log_fft), n = len(log_fft)*zero_pad_multiplier)

                #nicely plotable one-sided cestrum
                half_length = len(cepstrum)//2
                half_ceps = cepstrum[:half_length]
                quefrency = np.linspace(0, 1/2, half_length) #1/2 is because the fft is over one second

                #compute rpm
                T = quefrency[np.argmax(half_ceps)]
                rpm.append(60/T)

            else:
                counter += 1


        #Housekeeping
        capture.release() 
        cv2.destroyAllWindows()

        return rpm, half_ceps, quefrency, var, marker_outline

