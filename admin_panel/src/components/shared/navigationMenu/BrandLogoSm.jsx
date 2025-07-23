// BrandLogoSm.jsx
export default function BrandLogoSm({ className = "" }) {
  return (
    <svg
      className={className}
      width="40"
      height="40"
      viewBox="0 0 40 40"
      role="img"
      aria-label="EP"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect width="40" height="40" fill="#fff" />
      {/* “EP” monogram */}
      <text
        x="8.5" y="26"
        fontFamily="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"
        fontWeight="800"
        fontSize="18"
        letterSpacing="0.6"
        fill="#111"
      >
        EP
      </text>
    </svg>
  );
}
