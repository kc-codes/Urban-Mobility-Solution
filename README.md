
# Urban Mobility Solution

Integrated real-time data analytics for
optimized public transport, innovative road
monitoring using demand prediction, and conditioning tech for
sustainability, real time pothole detection either by image or video, smart parking count system for
efficiency using AI/ML.



## Features

- Demand prediction
- Pothole Detetection 
- Real time road damage detection
- Live previews
- Cross platform


## Installation
To install and run this project you would need to do the following:

1. Clone the repo: 
```bash
git clone https://github.com/yourusername/Urban-Mobility-Solution.git
```
2. Navigate into the project directory: 
```bash
cd Urban-Mobility-Solution
```
3. Install dependencies: 
```bash 
pip install -r requirements.txt

```
4. Run the project: 
```bash
  streamlit run app.py
```
    
5. You can now view the app in your browser.

  Local URL: http://localhost:8501 

6. To close the server press `ctrl + c`

## Usage/Examples

After completing the Installation steps the app can be used.

- How to use the app ?
### `For using Demand Prediction`
1. Navigate to the Demand Prediction section.
2. Click Browse Files
3. Upload a .csv file { For Testing .csv file included inside the directory named train_revised.csv }
4. After uploading .csv into the model the app will display the Home page of Deman Prediction below the video section.
5. You can select the appropriate demand page from the Choose deman page dropbox section.

### `For using Road Damage Assessment`
1. Navigate to the Image section.
2. Click Browse File.
3. Upload a JPG or JPEG file { For Testing .jpg file included inside the directory named Pothole.jpg }
4. After Uploading the file model will process the file and Provide you the approptiate output.

### `For using Real Time Road Damage Assesment`
1. Navigate to the Video section.
2. Click Browse File.
3. Upload a MP4 or MPEG4 file { For Testing .mp4 file included inside the home directory named assample_video.mp4 }
4. After Uploading the file, model will process the file in real time and a new processing window will open in the taskbar.
` Note: Processing my take time, depends on the speed of the processor `

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
## FAQ

1.  #### If encountered any error in terminal while running. What to Do ???

Check if all libraries are installed properly if not run `pip install -r requirements.txt` again.


## Authors

- [@Kunal Chaudhari](https://github.com/kc-codes)
- [@Karan Sankhe](https://github.com/Karansankhe)