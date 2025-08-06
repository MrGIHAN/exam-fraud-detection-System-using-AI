import { Navigate } from "react-router-dom";
import { getToken } from "./token";

const PrivateRoute = ({ children }) => {
  const token = getToken();
  return token ? children : <Navigate to="/authentication/login/creative" />;
};

export default PrivateRoute;
