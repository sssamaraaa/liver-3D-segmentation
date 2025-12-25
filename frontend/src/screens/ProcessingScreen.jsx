import AppLayout from "../app/AppLayout";

export default function ProcessingScreen({ progress }) {
  return (
    <AppLayout>
      <div className="processing-screen">
        <div className="processing-state">
            <h2>Обработка файла...</h2>
            <div className="spinner"></div>

            <div className="progress-wrapper">
            <div className="progress-bar" style={{ width: `${progress}%` }} />
            <p>{progress}%</p>
            </div>

            <p>Это может занять несколько минут</p>
        </div>
      </div>
    </AppLayout>
  );
}
