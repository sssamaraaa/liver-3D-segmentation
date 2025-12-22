import { useEffect, useRef, useState } from "react";

export default function SliceViewer({ mask }) {
  const canvasRef = useRef();
  const [slice, setSlice] = useState(0);

  useEffect(() => {
    if (!mask) return;

    fetch(mask[slice])
      .then(r => r.arrayBuffer())
      .then(buf => {
        const data = new Uint8Array(buf);
        const ctx = canvasRef.current.getContext("2d");
        const img = ctx.createImageData(256, 256);

        for (let i = 0; i < data.length; i++) {
          img.data[i * 4 + 1] = data[i] ? 255 : 0;
          img.data[i * 4 + 3] = 255;
        }

        ctx.putImageData(img, 0, 0);
      });
  }, [slice, mask]);

  return (
    <div className="flex flex-col items-center">
      <canvas ref={canvasRef} width={256} height={256} />
      <input
        type="range"
        min={0}
        max={mask.length - 1}
        value={slice}
        onChange={e => setSlice(+e.target.value)}
      />
    </div>
  );
}
