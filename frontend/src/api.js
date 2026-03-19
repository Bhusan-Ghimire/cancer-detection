import axios from "axios";

const host = window.location.hostname;
const MIDDLEWARE_URL = import.meta.env.VITE_MIDDLEWARE_URL || `http://${host}:3000`;
export async function predictImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post(`${MIDDLEWARE_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
    });

    return response.data;
}
