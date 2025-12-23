export default function SliceCanvas({ image }) {
  return (
    <div className="flex-1 flex items-center justify-center bg-black">
      <img
        src={`data:image/png;base64,${image}`}
        alt="slice"
        className="max-h-full max-w-full object-contain"
      />
    </div>
  );
}
