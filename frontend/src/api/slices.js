export async function fetchSliceStack(maskPath, axis) {
  const res = await fetch("http://localhost:8000/slices/stack", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mask_path: maskPath,
      axis,
    }),
  });

  if (!res.ok) {
    throw new Error("Failed to load slices");
  }

  return await res.json();
}
