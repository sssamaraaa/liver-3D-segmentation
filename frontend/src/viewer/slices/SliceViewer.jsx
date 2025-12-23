import { useEffect, useState } from "react";
import { fetchOverlay } from "/src/api/slices.js";

export default function SliceViewer({ ctPath, maskPath, axis }) {
  const [slices, setSlices] = useState([]);
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!ctPath || !maskPath) return;

        fetchOverlay(ctPath, maskPath, axis)
          .then(data => {
            setSlices(data.slices);
            setIndex(Math.floor(data.slices.length / 2));
          })
          .catch(console.error);
      }, [ctPath, maskPath, axis]);

      if (!slices.length) {
        return <div style={{ color: "white" }}>Загрузка срезов…</div>;
      }

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "black",
        display: "flex",
        flexDirection: "column",
        alignItems: "center"
      }}
    >
      <img
        src={`data:image/png;base64,${slices[index]}`}
        alt="slice"
        style={{
          maxHeight: "80vh",
          imageRendering: "pixelated"
        }}
      />

      <input
        type="range"
        min={0}
        max={slices.length - 1}
        value={index}
        onChange={e => setIndex(Number(e.target.value))}
        style={{ width: "80%", marginTop: 10 }}
      />

      <div style={{ color: "white" }}>
        {axis} slice {index + 1} / {slices.length}
      </div>
    </div>
  );
}
