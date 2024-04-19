use dasp::{interpolate::sinc::Sinc, ring_buffer, signal, Sample, Signal};
use hound::{WavReader, WavWriter};
use std::path::PathBuf;

pub struct AudioResampler {
    input_path: PathBuf,
    target_sample_rate: u32,
}

impl AudioResampler {
    pub fn new(input_path: PathBuf, target_sample_rate: u32) -> Self {
        AudioResampler {
            input_path,
            target_sample_rate,
        }
    }

    pub fn resample(&self) -> Result<String, hound::Error> {
        // Try to open the WAV file
        let reader = WavReader::open(&self.input_path)?;
        println!("reader successful");

        // Get the WAV spec and create a target with the new desired sample rate
        let spec = reader.spec();
        let mut target = spec;
        target.sample_rate = self.target_sample_rate;

        // Read the interleaved samples and convert them to a signal
        let samples = reader
            .into_samples()
            .filter_map(Result::ok)
            .map(i16::to_sample::<f64>);
        let signal = signal::from_interleaved_samples_iter(samples);

        // Convert the signal's sample rate using `Sinc` interpolation
        let ring_buffer = ring_buffer::Fixed::from([[0.0]; 100]);
        let sinc = Sinc::new(ring_buffer);
        let new_signal =
            signal.from_hz_to_hz(sinc, spec.sample_rate as f64, target.sample_rate as f64);

        // Generate a new output path by appending "_resampled.wav" to the input file name
        let output_path = self
            .input_path
            .with_file_name(self.input_path.file_name().unwrap().to_str().unwrap().to_owned()
                + "_resampled.wav");

        // Write the result to the output file
        let mut writer = WavWriter::create(&output_path, target)?;
        for frame in new_signal.until_exhausted() {
            writer.write_sample(frame[0].to_sample::<i16>())?;
        }

        Ok(output_path.to_str().unwrap().to_owned())
    }
}