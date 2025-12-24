import { useEffect, useState} from "react";
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
        <input
          type="range"
          min={0}
          max={slices.length - 1}
          value={index}
          onChange={e => setIndex(Number(e.target.value))}
          className="slice-slider"
        />
        
        <div className="slice-counter">
          {axis.toUpperCase()} срез {index + 1} / {slices.length}
        </div>

        {originalSize && (
          <div className="slice-resolution">
            Размер: {originalSize}px
          </div>
        )}

        {isLargeImage && (
          <div className="slice-warning">
            Изображение уменьшено для просмотра
          </div>
        )}
      </div>
    </div>
  );
}