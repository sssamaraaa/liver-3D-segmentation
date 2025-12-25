import { useEffect, useState } from "react";
import { fetchOverlay } from "/src/api/slices.js";

export default function SliceViewer({ ctPath, maskPath, axis }) {
  const [slices, setSlices] = useState([]);
  const [index, setIndex] = useState(0);
  const [originalSize, setOriginalSize] = useState("");

  useEffect(() => {
    if (!ctPath || !maskPath) return;

    fetchOverlay(ctPath, maskPath, axis)
      .then(data => {
        setSlices(data.slices);
        setIndex(Math.floor(data.slices.length / 2));
        if (data.image_size) {
          setOriginalSize(`${data.image_size[0]}×${data.image_size[1]}`);
        }
      })
      .catch(console.error);
  }, [ctPath, maskPath, axis]);

  useEffect(() => {
    if (slices.length && !originalSize) {
      const img = new Image();
      img.onload = () => {
        setOriginalSize(`${img.naturalWidth}×${img.naturalHeight}`);
      };
      img.src = `data:image/png;base64,${slices[index]}`;
    }
  }, [slices, index, originalSize]);

  const handleSliderChange = (e) => {
    setIndex(Number(e.target.value));
  };

  const handlePrev = () => {
    setIndex(prev => Math.max(0, prev - 1));
  };

  const handleNext = () => {
    setIndex(prev => Math.min(slices.length - 1, prev + 1));
  };

  if (!slices.length) {
    return (
      <div className="slice-viewer slice-viewer--loading">
        Загрузка срезов…
      </div>
    );
  }

  const isNonStandardSize = originalSize && !originalSize.includes('512×512');
  const isLargeImage = originalSize && originalSize.includes('1024');

  return (
    <div className="slice-viewer">
      <div className="slice-container">
        <img
          src={`data:image/png;base64,${slices[index]}`}
          alt="slice"
          className="slice-image"
        />
      </div>

      <div className="slice-controls">
        <div className="slice-navigation">
          <button 
            className="slice-nav-btn" 
            onClick={handlePrev}
            disabled={index === 0}
          >
            ←
          </button>
          
          <div className="slice-slider-container">
            <input
              type="range"
              min={0}
              max={slices.length - 1}
              value={index}
              onChange={handleSliderChange}
              className="slice-slider"
              style={{
                background: `linear-gradient(to right, #10B981 0%, #10B981 ${(index / (slices.length - 1)) * 100}%, #4b5563 ${(index / (slices.length - 1)) * 100}%, #4b5563 100%)`
              }}
            />
            <div className="slice-slider-ticks">
              <span>0</span>
              <span>{Math.floor((slices.length - 1) / 2)}</span>
              <span>{slices.length - 1}</span>
            </div>
          </div>
          
          <button 
            className="slice-nav-btn" 
            onClick={handleNext}
            disabled={index === slices.length - 1}
          >
            →
          </button>
        </div>
        
        <div className="slice-info">
          <div className="slice-counter">
            <span className="slice-counter-label">{axis.toUpperCase()}-срез:</span>
            <span className="slice-counter-value">{index + 1} / {slices.length}</span>
          </div>

          {originalSize && (
            <div className="slice-resolution">
              <span className="slice-resolution-label">Размер:</span>
              <span className="slice-resolution-value">{originalSize}px</span>
            </div>
          )}

          {isLargeImage && (
            <div className="slice-warning">
              Изображение уменьшено для просмотра
            </div>
          )}
        </div>
      </div>
    </div>
  );
}