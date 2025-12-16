export async function uploadSegmentation(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://localhost:8000/segmentation/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Segmentation failed");
  }

  return await response.json();
}
