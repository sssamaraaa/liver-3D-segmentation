export default function AppLayout({ children }) {
  return (
    <div className="app-container">

      <div className="upload-frame">
        <h1 className="upload-title">Сегментация КТ печени</h1>
        <h5 className="upload-title-h5">3D просмотр и анализ</h5>
      </div>

      <div style={{ height: "calc(100vh - 64px)" }}>
        {children}
      </div>

    </div>
  );
}