export async function fetchOverlay(ctPath, maskPath, axis) {
  const res = await fetch("/slices/overlay", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ct_path: ctPath,
      mask_path: maskPath,
      axis,
    }),
  });

  if (!res.ok) {
    throw new Error("Overlay request failed");
  }

  return res.json();
}
