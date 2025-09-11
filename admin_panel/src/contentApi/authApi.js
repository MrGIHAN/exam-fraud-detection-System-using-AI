import axios from 'axios';
import BASE_URL from '../config/apiConfig';



export const loginUser = async (email, password) => {
  const res = await axios.post(`${BASE_URL}/teacher/login`, { email, password });
  return res.data;
};
