export async function runSegmentation(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://localhost:8000/segmentation/predict", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorText = await res.text();
    console.error('Server error:', errorText);
    throw new Error(`Segmentation failed: ${res.status} ${errorText}`);
  }

  return await res.json();
}