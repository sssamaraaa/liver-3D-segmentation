export async function buildMesh(maskPath) {
  const response = await fetch("http://localhost:8000/mesh/build", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      mask_path: maskPath,
    }),
  });

  if (!response.ok) {
    throw new Error("Mesh build failed");
  }

  return await response.json();
}
