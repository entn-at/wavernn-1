import matplotlib.pyplot as plt
import librosa
import numpy as np

"""
We are going to use librosa to parse and render audio files

"""

class preprocessor:

    def __init__(self):
        self.sample_rate=16000
        self.input_data=None
        self.bit_rate=np.int16
        self.bit_width=np.iinfo(self.bit_rate).max
        self.split_rate=np.uint8
        self.split_width=np.iinfo(self.split_rate).max


    def import_audio_file(self,file_name):
        self.input_data,_=librosa.load(file_name,sr=self.sample_rate,mono=True)
        #Rescaling it to 16-bit data
        # plt.plot(self.input_data)
        self.input_data+=1
        # plt.plot(self.input_data)
        self.input_data*=self.bit_width
        # plt.plot(self.input_data)


    def fetch_coarse_and_fine_parameters(self):

    	self.coarse_data, self.fine_data=np.divmod(self.input_data,self.split_width)



    def export_audio_file(self,file_name):
    	
    	try:
    		self.input_data/=self.bit_width
    		librosa.output.write_wav(file_name,self.input_data,sr=self.sample_rate)
    		return True
    	except Exception as e:
    		print(e)
    		return False

if __name__=="__main__":
    pre=preprocessor()
    pre.import_audio_file("tt.wav")
    pre.fetch_coarse_and_fine_parameters()
    pre.export_audio_file("temp1.wav")