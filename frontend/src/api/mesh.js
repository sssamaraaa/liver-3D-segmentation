export async function buildMesh(maskPath) {
  console.log('Building mesh for:', maskPath);
  
  const res = await fetch("/mesh/build", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      mask_path: maskPath,
      smooth_iter: 30,
      decimate_ratio: 0.5,
    }),
  });

  if (!res.ok) {
    const errorText = await res.text();
    console.error('Mesh error:', errorText);
    throw new Error(`Mesh build failed: ${res.status} ${errorText}`);
  }

  return await res.json();
}