import numpy as np
import cv2
import os
from scipy.signal import find_peaks



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
        frames_in_mem = []
        ROI_flag = False
        while True: 
            #load frames
            ret, frame = capture.read()

            #load frames in grayscale in memory
            if ret:     
                frames_in_mem.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) 

            #compute ROI
            if not ROI_flag and len(frames_in_mem) == fps:
                centroid_x, centroid_y, var, marker_outline = self.compute_ROI(frames_in_mem)
                ROI_flag = True

            #compute FFT
            if len(frames_in_mem) == fps: 
                omega, half_ceps, quefrency = self.compute_spectrum(centroid_x, centroid_y, fps, frames_in_mem)
                rpm.append(omega)
                frames_in_mem = []

            #end condition
            if not ret:
                break

        #Housekeeping
        capture.release() 
        cv2.destroyAllWindows()


        return rpm, half_ceps, quefrency, var, marker_outline




    def compute_ROI(self, frames): 
        '''
        compute the region of interest from analyzing variance
        '''
        #convert to np.array
        frames = np.array(frames, dtype=np.uint8)

        #compute threshold
        var = np.var(frames, axis = 0)
        var_norm = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(var_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, labels, _, _ = cv2.connectedComponentsWithStats(thresh)

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


        return centroid_x, centroid_y, var, marker_outline
        


    def compute_spectrum(self, centroid_x, centroid_y, fps, frames): 
        '''
        compute the spectrum from a group of frames
        '''
        #convert to np.array
        frames = np.array(frames, dtype=np.uint8)

        #obtain ROI
        ROI = frames[:, centroid_y, centroid_x].T # shape: (N_ROI, fps)

        #compute fft
        hanning = np.hanning(fps)
        zero_pad_multiplier = 2
        windowed_fft = (ROI.T - np.mean(ROI, axis = -1)).T*hanning
        fft = np.fft.rfft(windowed_fft, axis = -1, n = fps*zero_pad_multiplier)
        abs_fft = np.squeeze(np.mean(np.abs(fft), axis = 0))
        freqs = np.fft.rfftfreq(n = fps*zero_pad_multiplier, d = 1/fps)

        #fit for frequency
        rpm_idx = np.argmax(abs_fft)
        y1 = abs_fft[rpm_idx - 1]
        y2 = abs_fft[rpm_idx]
        y3 = abs_fft[rpm_idx + 1]
        df = freqs[1] - freqs[0]
        b = (y3-y1)/(2*df)
        a = (y1 - 2*y2 + y3)/(2*df**2)
        true_freq = freqs[rpm_idx] - b/(2*a)
        rpm = 60*true_freq


        return rpm, abs_fft, freqs


